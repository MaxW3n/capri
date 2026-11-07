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
import json, math, os, queue, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from safetensors.numpy import load_file

from hooks import attach

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

PORT = 8777
MODEL = "mlx-community/Qwen3.8-27B-4bit"
VECTOR = "capri.safetensors"  # layer20: the vector, already scaled to layer 20's typical hidden-state size
LAYER = 20
SIZE = "27B"  # label shown in the chat header
SYSTEM = None  # the steering vector carries her voice; no prompt
HISTORY = 12  # messages kept per request
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
ALLOWED_ORIGINS = {f"http://{h}" for h in ALLOWED_HOSTS}
# the 3D page (web/3d) loads its scripts, three.js and the model from these folders only
STATIC_ROOTS = ("web/3d/", "web/vendor/three/", "capri_project_capricorn/")
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".json": "application/json", ".gltf": "model/gltf+json",
                ".bin": "application/octet-stream", ".png": "image/png", ".webp": "image/webp",
                ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8"}

model, tok = load(MODEL)
DEFAULT_DOSE = 0.74  # the dose the vector was trained at
MIN_DOSE, MAX_DOSE = 0.0, round(max(1.2, DEFAULT_DOSE * 1.5), 2)
vec = mx.array(load_file(VECTOR)[f"layer{LAYER}"]).astype(mx.bfloat16)
tap = attach(model, LAYER, vec=vec, dose=0.0)
sampler = make_sampler(temp=0.7, top_p=0.95)
EOS = sorted(set(tok.eos_token_ids))
ENDS = (".", "!", "?", '"', "'", ")", "…")


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
Q_TO_CAP = {tok.encode(q)[0]: tok.encode(c)[0] for q, c in (("Q", "Cap"), (" Q", " Cap"), ("q", "cap"), (" q", " cap"))}
CAP_IDS, RI = set(Q_TO_CAP.values()), tok.encode("Capri")[-1]
BLOCK = [tok.encode(" Alibaba")[0], tok.encode("Alibaba")[-1]]
TONG, YI = {tok.encode(" Tongyi")[0], tok.encode("Tongyi")[1]}, tok.encode(" Tongyi")[-1]


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
NAME_IDS = {i for w in ("Capri", " Capri", "capri", " capri") for i in tok.encode(w)}


def presence_penalty(tokens, logits):
    start = getattr(finish_sentences, "start", None) or tokens.size
    own = [int(t) for t in tokens[max(start, tokens.size - PRESENCE_WINDOW):].tolist() if int(t) not in NAME_IDS]
    if own:
        logits[..., mx.array(sorted(set(own)))] -= PRESENCE
    return logits


processors = [presence_penalty, finish_sentences, identity_guard]
jobs = queue.Queue()  # replies to generate, run one at a time on the main thread (the tap's dose is shared state)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")  # always re-check: page / script edits show on a plain reload
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path):
        """Serve a file from the 3D frontend, the vendored three.js or the Capri model; nothing else on disk."""
        rel = os.path.normpath(path.split("?", 1)[0].lstrip("/"))
        if rel.startswith("..") or os.path.isabs(rel) or not rel.startswith(STATIC_ROOTS):
            return False
        full = os.path.join(HERE, rel)
        if not os.path.isfile(full):
            return False
        ext = os.path.splitext(full)[1].lower()
        with open(full, "rb") as f:
            self._send(200, f.read(), STATIC_TYPES.get(ext, "application/octet-stream"))
        return True

    def do_GET(self):
        if self.path.split("?", 1)[0] in ("/", "/index.html", "/3d"):
            # the 3D scene; it asks /info for the dose range and model size
            self._send(200, open("web/3d/index.html", "rb").read(), "text/html; charset=utf-8")
        elif self._static(self.path):
            pass
        elif self.path in ("/classic", "/classic.html"):
            html = open("web/chat.html").read().replace('max="1.2"', f'max="{MAX_DOSE}"') \
                .replace('value="0.85"', f'value="{DEFAULT_DOSE}"').replace('>0.85<', f'>{DEFAULT_DOSE:.2f}<') \
                .replace('0.85 Capri', f'{DEFAULT_DOSE:.2f} Capri').replace('steered 4B', f'steered {SIZE}') \
                .replace('at layer 10', f'at layer {LAYER}')
            self._send(200, html.encode(), "text/html; charset=utf-8")
        elif self.path == "/info":
            self._send(200, json.dumps({"layer": LAYER, "min_dose": MIN_DOSE, "max_dose": MAX_DOSE,
                                        "default_dose": DEFAULT_DOSE, "size": SIZE}).encode(),
                       "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/chat":
            return self._send(404, b"not found", "text/plain")
        # only this page may call /chat: blocks other websites' no-cors POSTs and DNS rebinding
        host_ok = self.headers.get("Host") in ALLOWED_HOSTS
        origin = self.headers.get("Origin")
        json_ok = self.headers.get("Content-Type", "").startswith("application/json")
        if not host_ok or (origin and origin not in ALLOWED_ORIGINS) or not json_ok:
            return self._send(403, b'{"error": "forbidden"}', "application/json")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            msgs = [m for m in req["messages"][-HISTORY:]
                    if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"]]
            if not msgs or msgs[-1]["role"] != "user":
                raise ValueError("the conversation must end with a user message")
            dose = float(req.get("dose", DEFAULT_DOSE))
            if not math.isfinite(dose):
                raise ValueError("dose must be a number")
            dose = min(max(dose, MIN_DOSE), MAX_DOSE)
            prompt = tok.apply_chat_template(([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + msgs,
                                             add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except Exception as e:
            return self._send(400, json.dumps({"error": f"bad request: {e}"}).encode(), "application/json")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        # generation runs on the main thread (MLX's GPU stream belongs to the thread that loaded the model); this
        # request thread queues the job and streams back whatever the main thread puts on its own queue
        out, stop = queue.Queue(), threading.Event()
        jobs.put((prompt, dose, out, stop))
        try:
            while True:
                ev = out.get()
                if ev is None:
                    break
                emit(ev)
        except (BrokenPipeError, ConnectionResetError):
            stop.set()  # the page stopped listening: the main thread drops the rest


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


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # a browser dropping a connection mid-reply isn't an error
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


if __name__ == "__main__":
    print(f"Capri on http://127.0.0.1:{PORT}  (layer {LAYER}, dose {MIN_DOSE} to {MAX_DOSE})", flush=True)
    # files and requests are served from threads (the page fetches dozens at once); every reply is generated here,
    # on the main thread, one at a time
    srv = Server(("127.0.0.1", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        while True:
            generate(*jobs.get())
    except KeyboardInterrupt:
        srv.shutdown()
