"""Chat with Capri in the terminal: stock Qwen3.8-27B (4-bit, MLX) plus one steering vector at layer 20.

    pip install mlx-lm safetensors huggingface_hub
    python capri_chat.py               # dose 0.74
    python capri_chat.py --dose 0.5    # milder

Needs an Apple Silicon Mac with about 20 GB of free memory.
"""
import argparse

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from safetensors.numpy import load_file

BASE, REPO, LAYER = "mlx-community/Qwen3.8-27B-4bit", "MaxW3n/capri-steering", 20
ap = argparse.ArgumentParser()
ap.add_argument("--dose", type=float, default=0.74)
ap.add_argument("--vector", default="capri.safetensors" if __import__("os").path.exists("capri.safetensors") else None,
                help="local path to capri.safetensors (default: ./capri.safetensors, else download from the Hub)")
args = ap.parse_args()

model, tok = load(BASE)
vec = mx.array(load_file(args.vector or hf_hub_download(REPO, "capri.safetensors"))["layer20"]).astype(mx.bfloat16)


class Steer:
    """Wraps layer 20: run the real layer, then add dose * vector to its output for every token."""

    def __init__(self, layer):
        self.layer, self.is_linear = layer, layer.is_linear

    def __call__(self, x, mask=None, cache=None):
        return self.layer(x, mask=mask, cache=cache) + args.dose * vec


layers = model.language_model.model.layers
layers[LAYER] = Steer(layers[LAYER])

# Stopgap: the base model sometimes calls itself "Qwen". "Qwen" starts with a "Q" token, so that token's
# probability moves onto "Cap" and the next token is forced to "ri", giving "Capri". "Alibaba" is blocked.
Q_TO_CAP = {tok.encode(q)[0]: tok.encode(c)[0] for q, c in (("Q", "Cap"), (" Q", " Cap"), ("q", "cap"), (" q", " cap"))}
RI, BLOCK = tok.encode("Capri")[-1], [tok.encode(" Alibaba")[0], tok.encode("Alibaba")[-1]]
state = {"swapped": False, "start": 0}


def identity_guard(tokens, logits):
    if tokens.size > state["start"] and int(tokens[-1]) in Q_TO_CAP.values() and state["swapped"]:
        state["swapped"] = False
        logits[..., :] = -float("inf")
        logits[..., RI] = 0.0
        return logits
    state["swapped"] = False
    for q, c in Q_TO_CAP.items():
        if (logits[..., q] > logits[..., c]).all():
            state["swapped"] = True
        logits[..., c] = mx.logaddexp(logits[..., q], logits[..., c])
        logits[..., q] = -float("inf")
    logits[..., mx.array(BLOCK)] = -float("inf")
    return logits


NAME_IDS = {i for w in ("Capri", " Capri", "capri", " capri") for i in tok.encode(w)}


def presence_penalty(tokens, logits):
    """Stops "Hey! Hey! Hey!" loops: tokens among her last 20 in this reply lose 1.5 logits (her name never does)."""
    own = sorted({int(t) for t in tokens[max(state["start"], tokens.size - 20):].tolist()} - NAME_IDS)
    if own:
        logits[..., mx.array(own)] -= 1.5
    return logits


sampler, history = make_sampler(temp=0.7, top_p=0.95), []
print(f"Capri (dose {args.dose}). Ctrl-C to quit.\n")
while True:
    try:
        line = input("you> ")
    except (EOFError, KeyboardInterrupt):
        break
    history.append({"role": "user", "content": line})
    prompt = tok.apply_chat_template(history[-12:], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    state["start"] = len(tok.encode(prompt))
    reply = ""
    print("capri> ", end="", flush=True)
    for r in stream_generate(model, tok, prompt, max_tokens=200, sampler=sampler,
                             logits_processors=[presence_penalty, identity_guard]):
        print(r.text, end="", flush=True)
        reply += r.text
    print("\n")
    history.append({"role": "assistant", "content": reply.strip()})
