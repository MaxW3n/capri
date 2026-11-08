"""Local Capri chat: Qwen3.8-27B (4-bit, MLX) with no system prompt, steered by one vector at layer 20.

    pip install mlx-lm safetensors
    python server.py          # then open http://127.0.0.1:8777

Her voice comes from capri.safetensors: 5,120 numbers added to the model's hidden state after layer 20 on every
token (h = h + dose * vector). Dose 0 is the stock model.

GET / is the 3D frontend (web/3d: Capri on a rooftop in a neon city, replies in a speech bubble);
GET /classic is the flat chat page (web/chat.html).

POST /chat {"messages": [{role, content}, ...], "dose": 0.74} streams the reply as server-sent events. Every request
re-reads the whole conversation, so a new dose applies cleanly to the next reply.
"""
import math, os, queue, sys, threading

from local_http import LocalServer, ModelState, PORT

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "mlx-community/Qwen3.8-27B-4bit"
MODEL_REVISION = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
VECTOR = "capri.safetensors"  # layer20: the vector, already scaled to layer 20's typical hidden-state size
LAYER = 20
SIZE = "27B"  # label shown in the chat header
SYSTEM = None  # the steering vector carries her voice; no prompt
HISTORY = 12  # messages kept per request
DEFAULT_DOSE = 0.74  # the dose the vector was trained at
MIN_DOSE, MAX_DOSE = 0.0, round(max(1.2, DEFAULT_DOSE * 1.5), 2)
ENDS = (".", "!", "?", '"', "'", ")", "…")

# Importing this file does not import MLX or download/load a model. The HTTP
# listener starts first; the main thread subsequently owns both load and GPU work.
model = tok = tap = sampler = mx = stream_generate = None
EOS = []
Q_TO_CAP, CAP_IDS, RI, BLOCK, TONG, YI, NAME_IDS = {}, set(), None, [], set(), None, set()


def initialise_model(state):
    global model, tok, tap, sampler, mx, stream_generate, EOS
    global Q_TO_CAP, CAP_IDS, RI, BLOCK, TONG, YI, NAME_IDS

    local_path = os.environ.get("CAPRI_MODEL_PATH")
    if local_path:
        if not os.path.isdir(local_path):
            raise ValueError("CAPRI_MODEL_PATH is not a model directory")
        model_path = local_path
    else:
        state.set("loading", "Checking or downloading Capri's model. First setup downloads about 16 GB; progress is shown in Terminal.")
        print(state.snapshot()["message"], flush=True)
        from huggingface_hub import snapshot_download
        download = dict(repo_id=MODEL, revision=MODEL_REVISION,
                        allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "*.jinja"])
        try:
            model_path = snapshot_download(**download, local_files_only=True)
        except Exception:
            # Some Hub cache versions need fresh metadata even when weight
            # blobs already exist. The normal downloader reuses those blobs.
            model_path = snapshot_download(**download)

    state.set("loading", "Loading Capri's model into memory…")
    print(state.snapshot()["message"], flush=True)
    import mlx.core as mlx_core
    from mlx_lm import load, stream_generate as mlx_stream_generate
    from mlx_lm.sample_utils import make_sampler
    from safetensors.numpy import load_file
    from hooks import attach

    mx, stream_generate = mlx_core, mlx_stream_generate
    model, tok = load(model_path)
    vec = mx.array(load_file(os.path.join(HERE, VECTOR))[f"layer{LAYER}"]).astype(mx.bfloat16)
    tap = attach(model, LAYER, vec=vec, dose=0.0)
    sampler = make_sampler(temp=0.7, top_p=0.95)
    EOS = sorted(set(tok.eos_token_ids))
    Q_TO_CAP = {tok.encode(q)[0]: tok.encode(c)[0] for q, c in (("Q", "Cap"), (" Q", " Cap"), ("q", "cap"), (" q", " cap"))}
    CAP_IDS, RI = set(Q_TO_CAP.values()), tok.encode("Capri")[-1]
    BLOCK = [tok.encode(" Alibaba")[0], tok.encode("Alibaba")[-1]]
    TONG, YI = {tok.encode(" Tongyi")[0], tok.encode("Tongyi")[1]}, tok.encode(" Tongyi")[-1]
    NAME_IDS = {i for w in ("Capri", " Capri", "capri", " capri") for i in tok.encode(w)}


def finish_sentences(tokens, logits):
    """Block the end-of-reply token until the reply ends a sentence (she sometimes stopped mid-word),
    with a cap of 120 reply tokens so it can never run on."""
    gen = getattr(finish_sentences, "start", None)
    n = tokens.size - (gen or 0)
    if n < 120:
        tail = tok.decode(tokens[-4:].tolist()).rstrip()
        if not tail.endswith(ENDS):
            logits[..., EOS] = -float("inf")
    return logits


# Identity guard (until the unlikelihood retrain lands): "Qwen" is two tokens, "Q" + "wen". A bare "Q" / " Q" / "q" /
# " q" at the start of a word is almost always the start of "Qwen" in her replies, so its probability is moved onto
# the matching "Cap" token, and a "Cap" produced that way is forced on to "ri" ("Capri"). "Alibaba" and the "yi" of
# "Tongyi" are blocked outright. This hides the base model's identity; it doesn't change what the vector learned.


def identity_guard(tokens, logits):
    gen = getattr(finish_sentences, "start", None) or 0
    last = int(tokens[-1]) if tokens.size > gen else None
    if last in CAP_IDS and getattr(identity_guard, "swapped", False):
        logits[..., :] = -float("inf"); logits[..., RI] = 0.0  # finish the word: Cap -> Capri
        identity_guard.swapped = False
        return logits
    identity_guard.swapped = False
    for q, c in Q_TO_CAP.items():
        lq, lc = logits[..., q], logits[..., c]
        if (lq > lc).all():
            identity_guard.swapped = True  # the swapped-in Cap carries the Q mass; if sampled, force "ri" next
        logits[..., c] = mx.logaddexp(lq, lc)
        logits[..., q] = -float("inf")
    logits[..., mx.array(BLOCK)] = -float("inf")
    if last in TONG:
        logits[..., YI] = -float("inf")
    return logits


# Presence penalty (stops "Hey! Hey! Hey!" loops): every token among the last 20 she has generated in THIS reply
# loses 1.5 logits (x0.22 odds). mlx-lm's built-in version also counted the prompt, so your words (and "Capri" when
# you typed it) were pushed down too; here the prompt never counts, and her name's tokens are never penalised.
PRESENCE, PRESENCE_WINDOW = 1.5, 20


def presence_penalty(tokens, logits):
    start = getattr(finish_sentences, "start", None) or tokens.size
    own = [int(t) for t in tokens[max(start, tokens.size - PRESENCE_WINDOW):].tolist() if int(t) not in NAME_IDS]
    if own:
        logits[..., mx.array(sorted(set(own)))] -= PRESENCE
    return logits


processors = [presence_penalty, finish_sentences, identity_guard]
jobs = queue.Queue()  # replies to generate, run one at a time on the main thread (the tap's dose is shared state)


def queue_reply(req):
    """Validate a chat request, then queue GPU work for the main thread."""
    raw = req.get("messages")
    if not isinstance(raw, list):
        raise ValueError("messages must be an array")
    msgs = [m for m in raw[-HISTORY:]
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str) and m["content"]]
    if not msgs or msgs[-1]["role"] != "user":
        raise ValueError("the conversation must end with a user message")
    dose = float(req.get("dose", DEFAULT_DOSE))
    if not math.isfinite(dose):
        raise ValueError("dose must be a number")
    dose = min(max(dose, MIN_DOSE), MAX_DOSE)
    prompt = tok.apply_chat_template(([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + msgs,
                                     add_generation_prompt=True, tokenize=False, enable_thinking=False)
    out, stop = queue.Queue(), threading.Event()
    jobs.put((prompt, dose, out, stop))
    return out, stop


def generate(prompt, dose, out, stop):
    """One reply, on the main thread. Events go to `out`; None ends the stream."""
    tap.dose = dose
    r = None
    try:
        finish_sentences.start = len(tok.encode(prompt))
        for r in stream_generate(model, tok, prompt, max_tokens=300, sampler=sampler, logits_processors=processors):
            if stop.is_set():
                break
            if r.text:
                out.put({"t": r.text})
        why = getattr(r, "finish_reason", None)
        print(f"done: {why}, {r.generation_tokens if r else 0} tokens, last token {getattr(r, 'token', None)}", flush=True)
        out.put({"done": True, "tps": round(r.generation_tps, 1) if r else 0, "finish": why,
                 "tokens": r.generation_tokens if r else 0})
    except Exception as e:
        out.put({"error": f"generation failed: {e}"})
    finally:
        out.put(None)


def run(*, initializer=None, stop_event=None, port=PORT, on_listening=None):
    """Start loopback HTTP before initialisation; keep all model work on this thread.

    The injected initializer/port/event are for model-free integration tests.
    The public command always uses port 8777 and initialise_model.
    """
    state = ModelState(layer=LAYER, size=SIZE, min_dose=MIN_DOSE,
                       max_dose=MAX_DOSE, default_dose=DEFAULT_DOSE)
    stop_event = stop_event or threading.Event()
    # Bind before importing MLX or obtaining weights: a port conflict must never
    # cause another 27B model to load.
    srv = LocalServer(state, queue_reply, root=HERE, port=port)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    print(f"Capri model runner listening on http://127.0.0.1:{srv.server_port}", flush=True)
    print("Open https://maxwen.ca/capri and choose Connect local Capri, or open the local address.", flush=True)
    try:
        if on_listening:
            on_listening(srv)
        try:
            (initializer or initialise_model)(state)
        except Exception as error:
            message = f"Capri could not load its model: {error}. Check Terminal, fix the problem, then restart the runner."
            state.set("error", message)
            print(message, file=sys.stderr, flush=True)
        else:
            state.set("ready", "Capri is ready.")
            print(f"Capri is ready (layer {LAYER}, dose {DEFAULT_DOSE}). Ctrl-C stops the runner.", flush=True)
        while not stop_event.is_set():
            try:
                job = jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            generate(*job)
    except KeyboardInterrupt:
        print("Stopping Capri…", flush=True)
    finally:
        state.set("error", "Capri's model runner has stopped.")
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    try:
        run()
    except OSError as error:
        print(f"Capri could not open 127.0.0.1:{PORT}: {error}. Stop the existing runner and try again.", file=sys.stderr)
        raise SystemExit(1)
