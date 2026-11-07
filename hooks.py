"""Read and steer Qwen3.5's residual stream in mlx-lm.

model.layers is a plain list, so we swap one entry for a wrapper that runs the
real layer, then records its output h (the residual stream after layer L) or
adds a steering vector to it.
"""
import mlx.core as mx


def layers_of(model):
    return model.language_model.model.layers


class Tap:
    """Wraps one decoder layer. record=True stores h; vec adds dose*vec to h."""

    def __init__(self, layer, record=False, vec=None, dose=0.0):
        self.layer = layer
        self.is_linear = layer.is_linear  # the model picks the mask with this
        self.record = record
        self.vec = vec
        self.dose = dose
        self.saved = []

    def __call__(self, x, mask=None, cache=None):
        h = self.layer(x, mask=mask, cache=cache)  # shape (batch, tokens, 2560)
        if self.vec is not None and self.dose:
            h = h + (self.dose * self.vec).astype(h.dtype)  # keep bf16; a float32 vec would upcast the rest
        if self.record:
            self.saved.append(h)
        return h


def attach(model, L, **kw):
    tap = Tap(layers_of(model)[L], **kw)
    layers_of(model)[L] = tap
    return tap


def detach(model, L):
    tap = layers_of(model)[L]
    if isinstance(tap, Tap):
        layers_of(model)[L] = tap.layer


def hidden_states(model, tokenizer, prompt, reply, L):
    """Prefill prompt+reply once; return h at layer L for the reply tokens only."""
    n_prompt = len(tokenizer.encode(prompt))
    ids = mx.array(tokenizer.encode(prompt + reply))[None]
    tap = attach(model, L, record=True)
    try:
        model(ids)  # one forward pass, no generation
        h = tap.saved[0][0, n_prompt:]  # drop batch dim and prompt positions
        mx.eval(h)
    finally:
        detach(model, L)
    return h  # (reply_tokens, 2560)
