// Adapted from Ruurd Kuiper's qwen38-27b-onnx-webgpu (Apache-2.0).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/ort.jspi.bundle.min.mjs';
import { AutoTokenizer, TextStreamer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';

const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/';
const MAX_CONTEXT_TOKENS = 4096;
const MAX_MESSAGE_CHARS = 32768;
let session, tokenizer, chatTemplate, config, graph, bindings, device, modelRoot, steering;
let releaseModelLock;
let activeRequestId = 0;
const report = { runtimeVersion: ort.env.versions.web, executionProviders: ['webgpu'], cpuFallbackDisabled: true, messages: [] };
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
function log(level, ...args) {
  const text = args.map(String).join(' ');
  report.messages.push({ level, text });
  if (level === 'info') self.postMessage({ type: 'progress', requestId: activeRequestId, message: text });
  else console[level](...args);
}
function heap() { return performance.memory ? { usedJSHeapSize: performance.memory.usedJSHeapSize } : null; }
async function fetchAsset(url, name, options) {
  let response;
  try {
    response = await fetch(url, { credentials: 'omit', referrerPolicy: 'no-referrer', ...options });
  } catch (error) {
    throw new Error(`Could not reach ${name}. Check your connection and retry.`, { cause: error });
  }
  if (!response.ok) throw new Error(`Could not load ${name} (HTTP ${response.status}). Try again later.`);
  return response;
}
async function json(url, name) { return (await fetchAsset(url, name)).json(); }
const modelUrl = file => `${modelRoot}/${encodeURIComponent(file)}`;
const gb = bytes => (bytes / 1e9).toFixed(2);

function reserveBrowserModel() {
  if (!navigator.locks) return;
  return new Promise((resolve, reject) => {
    // One 27B session per origin. Terminating the worker releases the lock.
    navigator.locks.request('capri-browser-model', { ifAvailable: true }, lock => {
      if (!lock) throw new Error('Capri is already loading or running in another tab. Unload it there or close that tab, then retry.');
      return new Promise(release => {
        releaseModelLock = release;
        resolve();
      });
    }).catch(reject);
  });
}

async function loadWeights(model, manifest) {
  let cache;
  try {
    cache = await caches.open('capri-27b-' + model.revision);
  } catch (error) {
    throw new Error('Chrome could not open model storage. Enable site data in a regular Chrome or Edge window and retry.', { cause: error });
  }
  const pinnedSizes = new Map(model.files.map(file => [file.name, file.bytes]));
  const manifestSizes = new Map(manifest.files.map(file => [file.name, file.bytes]));
  const shards = graph.externalData.map(path => {
    const bytes = pinnedSizes.get(path);
    if (!/^model\.onnx\.data_\d{3}$/.test(path) || !Number.isSafeInteger(bytes) || bytes <= 0 || bytes !== manifestSizes.get(path)) {
      throw new Error('Capri’s weight manifest does not match the pinned model. Reload the page and try again.');
    }
    return { path, bytes, url: modelUrl(path) };
  });
  const totalBytes = shards.reduce((sum, shard) => sum + shard.bytes, 0);
  report.totalWeightBytes = totalBytes;
  const externalData = new Array(shards.length);
  const missing = [];
  let next = 0, failure;
  async function cachedBlob(shard) {
    try {
      return await (await cache.match(shard.url))?.blob();
    } catch (error) {
      throw new Error(`Chrome could not read saved model file ${shard.path}. Reload the page and retry.`, { cause: error });
    }
  }
  function progress() {
    const saved = report.cachedWeightBytes + report.networkWeightBytes;
    const count = report.weightCacheHits + report.weightCacheMisses;
    log('info', `Saved ${gb(saved)} of ${gb(totalBytes)} GB (${count}/${shards.length} files)…`);
  }
  async function saveShard({ shard, index }) {
    const response = await fetchAsset(shard.url, `model file ${shard.path}`, { cache: 'no-store' });
    try {
      // Cache.put consumes the body before it commits the entry. Completed files survive a retry.
      await cache.put(shard.url, response);
    } catch (error) {
      if (error.name === 'QuotaExceededError') {
        throw new Error(`Chrome ran out of storage while saving ${shard.path}. Free disk space and retry. Completed files are saved.`, { cause: error });
      }
      throw new Error(`Could not save ${shard.path}. Check the connection and available disk space, then retry. Completed files are saved.`, { cause: error });
    }
    const data = await cachedBlob(shard);
    if (data?.size !== shard.bytes) {
      await cache.delete(shard.url);
      throw new Error(`The downloaded file ${shard.path} is incomplete. Retry to fetch the missing files.`);
    }
    externalData[index] = { path: shard.path, data };
    report.weightCacheMisses++;
    report.networkWeightBytes += data.size;
    progress();
  }
  async function loadShards() {
    log('info', `Checking ${gb(totalBytes)} GB of model weights in Chrome’s storage…`);
    for (const [index, shard] of shards.entries()) {
      const data = await cachedBlob(shard);
      if (data?.size === shard.bytes) {
        externalData[index] = { path: shard.path, data };
        report.weightCacheHits++;
        report.cachedWeightBytes += data.size;
      } else missing.push({ shard, index });
    }
    progress();
    if (!missing.length) return;
    // Chrome can report a synthetic quota. Cache.put reports actual storage failures.
    async function download() {
      try {
        while (!failure && next < missing.length) await saveShard(missing[next++]);
      } catch (error) {
        failure ??= error;
      }
    }
    await Promise.all([download(), download(), download()]);
    if (failure) throw failure;
  }
  if (navigator.locks) await navigator.locks.request('capri-27b-' + model.revision, loadShards);
  else await loadShards();
  return externalData;
}
// Match layer-specific names using the generated config, rather than guessing
// that every decoder layer has an ordinary key/value cache.
function cacheBindings() {
  const pairs = [
    ['past_key_names', 'present_key_names'], ['past_value_names', 'present_value_names'],
    ['past_conv_names', 'present_conv_names'], ['past_recurrent_names', 'present_recurrent_names'],
  ];
  const result = [];
  const decoder = config.model.decoder;
  for (const [past, present] of pairs) {
    if (!decoder.inputs[past]) continue;
    for (let layer = 0; layer < decoder.num_hidden_layers; layer++) {
      const input = decoder.inputs[past].replace('%d', layer);
      if (!session.inputNames.includes(input)) continue;
      const output = decoder.outputs[present].replace('%d', layer);
      if (!session.outputNames.includes(output)) throw new Error(`Missing output for ${input}: ${output}`);
      result.push({ input, output, metadata: graph.inputs.find((v) => v.name === input) });
    }
  }
  const supported = new Set([...result.map((v) => v.input), decoder.inputs.input_ids,
    decoder.inputs.attention_mask, decoder.inputs.position_ids, 'seqlens_k', 'total_sequence_length', 'steering_delta']);
  for (const name of session.inputNames) {
    if (!supported.has(name)) throw new Error(`Unimplemented graph input: ${name}`);
  }
  return result;
}

async function loadModel() {
  if (session) return;
  const loadStart = performance.now();
  if (typeof WebAssembly.Suspending !== 'function') throw new Error('This large model requires WebAssembly JSPI. Use current Chrome or Edge.');
  if (typeof TextStreamer !== 'function' || typeof TextStreamer.prototype.put !== 'function'
      || typeof TextStreamer.prototype.end !== 'function') {
    throw new Error('The browser tokenizer does not support text streaming.');
  }
  if (typeof Float16Array !== 'function') throw new Error('Your browser does not support FP16 model inputs.');
  report.status = 'loading';
  report.messages = [];
  Object.assign(report, { weightCacheHits: 0, weightCacheMisses: 0, cachedWeightBytes: 0, networkWeightBytes: 0 });
  log('info', 'Preparing WebGPU device…');
  if (!navigator.gpu) throw new Error('WebGPU is unavailable. Open Capri over HTTPS in current Chrome or Edge with graphics acceleration enabled.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('A hardware WebGPU adapter is required.');
  if (!adapter.features.has('shader-f16')) throw new Error('This FP16/INT4 export requires shader-f16.');
  await reserveBrowserModel();
  const requiredFeatures = ['shader-f16'];
  if (adapter.features.has('subgroups')) requiredFeatures.push('subgroups');
  const requiredLimits = {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
  const gpuDevice = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  device = gpuDevice;
  gpuDevice.addEventListener('uncapturederror', (e) => log('error', 'WebGPU:', e.error.message));
  gpuDevice.lost.then(info => {
    if (device === gpuDevice) {
      self.postMessage({ type: 'fatal', message: `Chrome lost the GPU device. Close other GPU-heavy tabs or apps and retry. ${info.message || ''}`.trim() });
    }
  });
  report.gpu = { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
    device: adapter.info.device, description: adapter.info.description,
    isFallbackAdapter: adapter.info.isFallbackAdapter,
    adapterFeatures: [...adapter.features], deviceFeatures: [...device.features],
    deviceLimits: requiredLimits };
  ort.env.webgpu.device = device;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = {
    wasm: ORT_DIST + 'ort-wasm-simd-threaded.jspi.wasm',
    mjs: ORT_DIST + 'ort-wasm-simd-threaded.jspi.mjs',
  };
  ort.env.logLevel = 'warning';

  const model = await json('./model.json', 'Capri’s model metadata');
  if (!/^[\w-]+\/[\w.-]+$/.test(model.repo) || !/^[a-f0-9]{40}$/.test(model.revision)) {
    throw new Error('Capri’s model metadata is missing its pinned Hugging Face revision. Reload the page and try again.');
  }
  modelRoot = `https://huggingface.co/${model.repo}/resolve/${model.revision}`;
  const manifest = await json('./capri.manifest.json', 'Capri’s model manifest');
  config = await json(modelUrl('genai_config.json'), 'the pinned model configuration');
  const vectorResponse = await fetchAsset('./steering.f32', 'Capri’s steering vector');
  steering = new Float32Array(await vectorResponse.arrayBuffer());
  if (steering.length !== 5120 || !steering.every(Number.isFinite) || !steering.some(v => v !== 0)) {
    throw new Error('Capri’s steering vector is invalid.');
  }
  const graphFile = manifest.browserGraph;
  if (graphFile !== 'capri.browser.onnx') throw new Error('Capri’s browser model graph is missing. Reload the page and try again.');
  graph = manifest.graphs.find((g) => g.file === graphFile);
  if (!graph?.externalData?.length) throw new Error('Capri’s browser model manifest is incomplete. Reload the page and try again.');
  if (manifest.graphs.some((g) => g.file.includes('mtp')) || config.model.mtp) throw new Error('Export only the decoder with exclude_mtp=true.');
  report.graphFile = graphFile;
  report.browserAdaptation = graphFile !== config.model.decoder.filename ? manifest.browserAdaptation : null;
  report.exportBytes = manifest.totalBytes;
  report.files = manifest.files;
  const downloadStart = performance.now();
  const externalData = await loadWeights(model, manifest);
  report.storagePersisted = await navigator.storage?.persisted?.().catch(() => false) ?? false;
  const response = await fetchAsset(`./${graph.file}`, 'Capri’s browser model graph');
  const modelBytes = new Uint8Array(await response.arrayBuffer());
  if (modelBytes.byteLength !== manifest.files.find(file => file.name === graphFile)?.bytes) {
    throw new Error('Capri’s browser model graph is incomplete. Reload the page and try again.');
  }
  report.graphAndWeightDownloadBytes = report.networkWeightBytes + modelBytes.byteLength;
  report.downloadMs = performance.now() - downloadStart;
  const preferredOutputLocation = Object.fromEntries(graph.outputs.map((v) =>
    [v.name, v.name === config.model.decoder.outputs.logits ? 'cpu' : 'gpu-buffer']));
  log('info', 'Creating WebGPU inference session…');
  const sessionStart = performance.now();
  try {
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['webgpu'], externalData, preferredOutputLocation,
      freeDimensionOverrides: { batch_size: 1 },
      // Explicitly reject fallback to WASM CPU inference.
      extra: { session: { disable_cpu_ep_fallback: '1' } },
      logSeverityLevel: 2,
    });
  } catch (error) {
    log('error', error);
    throw new Error('Capri could not start on this GPU. Close other GPU-heavy apps or tabs and retry. The weights are already saved.', { cause: error });
  }
  report.sessionCreateMs = performance.now() - sessionStart;
  bindings = cacheBindings();
  report.cacheTensorCount = bindings.length;
  report.graphInputs = session.inputNames;
  report.graphOutputs = session.outputNames;

  env.allowLocalModels = false;
  try {
    tokenizer = await AutoTokenizer.from_pretrained(model.repo, { revision: model.revision });
  } catch (error) {
    throw new Error('Could not load Capri’s tokenizer. Check your connection and retry. The weights are already saved.', { cause: error });
  }
  if (typeof tokenizer.decode !== 'function' || typeof tokenizer.apply_chat_template !== 'function'
      || !Array.isArray(tokenizer.all_special_ids)) {
    throw new Error('The browser tokenizer is missing methods or special token IDs.');
  }
  const templateResponse = await fetchAsset(modelUrl('chat_template.jinja'), 'the pinned chat template');
  chatTemplate = await templateResponse.text();
  report.loadMs = performance.now() - loadStart;
  report.modelResources = performance.getEntriesByType('resource')
    .filter((r) => r.name.startsWith(modelRoot + '/'))
    .map((r) => ({ name: new URL(r.name).pathname, decodedBodySize: r.decodedBodySize, transferSize: r.transferSize }));
  report.downloadedModelResourceBytes = report.modelResources.reduce((n, r) => n + r.decodedBodySize, 0);
  report.heapAfterLoad = heap();
  report.status = 'ready';
}

async function releaseModel() {
  const gpuDevice = device;
  device = undefined;
  try {
    await session?.release();
  } finally {
    session = tokenizer = undefined;
    gpuDevice?.destroy();
    releaseModelLock?.();
    releaseModelLock = undefined;
  }
}

function typedTensor(type, values, dims) {
  const constructors = { int64: BigInt64Array, int32: Int32Array, float16: Uint16Array,
    float: Float32Array, float32: Float32Array };
  const Constructor = constructors[type];
  if (!Constructor) throw new Error(`Unsupported graph tensor type: ${type}`);
  const data = typeof values === 'number' ? new Constructor(values) :
    Constructor.from(values, type === 'int64' ? BigInt : (v) => v);
  return new ort.Tensor(type === 'float' ? 'float32' : type, data, dims);
}
function initialState(metadata) {
  const symbols = { batch_size: 1, past_sequence_length: 0, kv_cache_dim: config.model.decoder.head_size };
  const dims = metadata.dims.map((v) => {
    if (typeof v === 'number') return v;
    if (v in symbols) return symbols[v];
    throw new Error(`Unresolved state dimension: ${v}`);
  });
  return typedTensor(metadata.type, dims.reduce((a, b) => a * b, 1), dims);
}
function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (exponent ? 2 ** (exponent - 15) * (1 + fraction / 1024) : 2 ** -14 * fraction / 1024);
}
async function sample(logits, temperature = 0) {
  const data = await logits.getData();
  const vocab = logits.dims.at(-1);
  const offset = data.length - vocab;
  let best = -Infinity, token = -1;
  const scores = new Float32Array(vocab);
  for (let i = 0; i < vocab; i++) {
    const value = logits.type === 'float16' && data instanceof Uint16Array ? halfToFloat(data[offset + i]) : data[offset + i];
    scores[i] = value;
    if (Number.isNaN(value)) throw new Error('NaN in logits.');
    if (value > best) { best = value; token = i; }
  }
  if (token < 0 || !Number.isFinite(best)) throw new Error('Invalid logits.');
  if (temperature > 0) {
    let total = 0;
    for (let i = 0; i < vocab; i++) total += scores[i] = Math.exp((scores[i] - best) / temperature);
    let target = Math.random() * total;
    for (let i = 0; i < vocab; i++) if ((target -= scores[i]) <= 0) return i;
  }
  return token;
}

async function generate({ messages, maxTokens, temperature, enableThinking, dose, requestId }, signal) {
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 1024) throw new Error("maxTokens must be between 1 and 1024.");
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) throw new Error("Invalid temperature.");
  if (!session || !tokenizer) throw new Error('Load the model first.');
  if (!Number.isFinite(dose) || dose < 0 || dose > 1.2) throw new Error('Invalid steering dose.');
  if (!Array.isArray(messages) || !messages.length || messages.length > 12 || messages.at(-1)?.role !== 'user') {
    throw new Error('Send 1 to 12 chat messages ending with a user message.');
  }
  if (!messages.every(m => (m?.role === 'user' || m?.role === 'assistant')
      && typeof m.content === 'string' && m.content.length > 0 && m.content.length <= MAX_MESSAGE_CHARS)) {
    throw new Error(`Chat messages need a user or assistant role and 1 to ${MAX_MESSAGE_CHARS} characters.`);
  }
  const chat = messages.map(({ role, content }) => ({ role, content }));
  const inputIds = signal.aborted ? [] : tokenizer.apply_chat_template(chat, {
    tokenize: true, return_tensor: false, return_dict: false,
    add_generation_prompt: true, enable_thinking: enableThinking, chat_template: chatTemplate,
  });
  if (!Array.isArray(inputIds) || !inputIds.every(Number.isInteger)) throw new Error('Unexpected tokenizer output.');
  const contextLimit = Math.min(config.model.context_length, MAX_CONTEXT_TOKENS);
  if (inputIds.length + maxTokens > contextLimit) throw new Error(`Prompt and reply exceed the ${contextLimit}-token browser limit.`);
  const decoder = config.model.decoder;
  const byName = Object.fromEntries(graph.inputs.map((v) => [v.name, v]));
  let state = {};
  let tokens = inputIds, pastLength = 0, firstTokenAt;
  const generated = [], steps = [];
  const start = performance.now();
  const eos = new Set([config.model.eos_token_id].flat());
  const decodeOptions = { skip_special_tokens: true, clean_up_tokenization_spaces: false };
  let streamed = '';
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    decode_kwargs: decodeOptions,
    callback_function(text) {
      if (signal.aborted) return;
      streamed += text;
      self.postMessage({ type: 'token', requestId, text });
    },
  });
  // Only sampled tokens go to the streamer. An empty first batch marks the skipped prompt.
  streamer.put([[]]);
  let delta;
  try {
    if (!signal.aborted) {
      for (const binding of bindings) state[binding.input] = initialState(binding.metadata);
      delta = new ort.Tensor('float16', Float16Array.from(steering, value => value * dose), [steering.length]);
    }
    for (let i = 0; i < maxTokens && !signal.aborted; i++) {
      const feeds = { ...state, steering_delta: delta };
      const ephemeral = [];
      const put = (name, values, dims) => {
        if (!name || !session.inputNames.includes(name)) return;
        const tensor = typedTensor(byName[name].type, values, dims);
        feeds[name] = tensor; ephemeral.push(tensor);
      };
      let outputs;
      try {
        put(decoder.inputs.input_ids, tokens, [1, tokens.length]);
        put('seqlens_k', [pastLength + tokens.length - 1], [1]);
        put('total_sequence_length', [pastLength + tokens.length], []);
        const positions = tokens.map((_, j) => pastLength + j);
        put(decoder.inputs.position_ids, [...positions, ...positions, ...positions], [3, 1, tokens.length]);
        outputs = await session.run(feeds);
        if (signal.aborted) break;
        const token = await sample(outputs[decoder.outputs.logits], temperature);
        if (signal.aborted) break;
        if (firstTokenAt === undefined) firstTokenAt = performance.now();
        generated.push(token);
        const nextState = {};
        for (const binding of bindings) {
          const value = outputs[binding.output];
          if (!value) throw new Error(`Missing state output: ${binding.output}`);
          if (value.location !== 'gpu-buffer') throw new Error(`State left GPU: ${binding.output} (${value.location})`);
          nextState[binding.input] = value;
        }
        steps.push({ step: i, inputTokens: tokens.length, pastLength, stateTensorsRetained: bindings.length,
          allStateOnGPU: true, kvLength: nextState[bindings.find((v) => v.input.startsWith('past_key_values.')).input].dims[2] });
        for (const tensor of Object.values(state)) tensor.dispose();
        state = nextState;
        pastLength += tokens.length;
        tokens = [token];
        if (eos.has(token)) break;
        streamer.put([[BigInt(token)]]);
      } finally {
        for (const tensor of ephemeral) tensor.dispose();
        const retained = new Set(Object.values(state));
        for (const tensor of Object.values(outputs ?? {})) if (!retained.has(tensor)) tensor.dispose();
      }
      await tick();
    }
    if (!signal.aborted) streamer.end();
    const end = performance.now();
    const finish = signal.aborted ? 'cancelled' : eos.has(generated.at(-1)) ? 'stop' : 'length';
    const result = { inputIds, generatedIds: generated, dose,
      promptTokens: inputIds.length, completionTokens: generated.length,
      text: signal.aborted || generated.length === 0 ? streamed : tokenizer.decode(generated, decodeOptions),
      finish, cancelled: signal.aborted,
      generatedTokens: generated.length, firstTokenMs: firstTokenAt === undefined ? null : firstTokenAt - start,
      tokensPerSecondAfterFirst: generated.length > 1 ? (generated.length - 1) * 1000 / (end - firstTokenAt) : null,
      elapsedMs: end - start, heap: heap(), steps };
    result.decodeTokensPerSecond = result.tokensPerSecondAfterFirst;
    report.lastGeneration = result;
    return result;
  } finally {
    delta?.dispose();
    for (const tensor of Object.values(state)) tensor.dispose();
  }
}

// Serialize requests so cache/session ownership cannot overlap.
let queue = Promise.resolve();
const generations = new Map();
self.onmessage = ({ data }) => {
  if (!Number.isSafeInteger(data?.requestId) || data.requestId < 1) return;
  if (data.type === 'cancel') {
    generations.get(data.requestId)?.abort();
    return;
  }
  if (generations.has(data.requestId)) return;
  const generation = data.type === 'generate' ? new AbortController() : null;
  if (generation) {
    if (generations.size >= 2) {
      self.postMessage({ type: 'error', requestId: data.requestId, error: 'A reply is already queued.' });
      return;
    }
    generations.set(data.requestId, generation);
  }
  queue = queue.then(async () => {
    activeRequestId = data.requestId;
    try {
      let value;
      if (data.type === 'load') {
        await loadModel();
        value = report;
      } else if (data.type === 'generate') {
        value = await generate(data, generation.signal);
      } else throw new Error('Unknown GenAI worker request.');
      self.postMessage({ type: 'result', requestId: data.requestId, value });
    } catch (error) {
      if (data.type === 'load') {
        report.status = 'error';
        await releaseModel().catch(cleanupError => log('error', cleanupError));
      }
      self.postMessage({ type: 'error', requestId: data.requestId, error: String(error) });
    } finally {
      if (generation) generations.delete(data.requestId);
    }
  });
};
