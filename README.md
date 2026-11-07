# Capri

Chat with Capri, a fan take on the character from Cyberpunk: Edgerunners 2, in a 3D neon city. She's stock
Qwen3.8-27B with no system prompt and no fine-tuned weights: one vector of 5,120 numbers is added to the model's
hidden state after layer 20 on every token, and that addition is her whole personality.

The vector is also on Hugging Face: [MaxW3n/capri-steering](https://huggingface.co/MaxW3n/capri-steering).

## Run it

Mac only: it uses MLX, Apple's machine-learning library, so it needs Apple Silicon and about 20 GB of free memory
(a 32 GB Mac or bigger).

```bash
git clone https://github.com/MaxW3n/capri
cd capri
pip install mlx-lm safetensors
python server.py
```

Open http://127.0.0.1:8777. The first start downloads the base model, `mlx-community/Qwen3.8-27B-4bit` (about 15 GB).
Replies run at roughly 15 tokens a second on an M4 Max.

Prefer a terminal? `python capri_chat.py` is the same chat without the 3D page.

## What's here

| Path | What it is |
|---|---|
| `capri.safetensors` | the steering vector (`layer20`), already scaled to layer 20's typical hidden-state size |
| `server.py` | loads the model, adds the vector at layer 20, streams replies to the page |
| `hooks.py` | the wrapper that adds the vector to layer 20's output |
| `capri_chat.py` | terminal chat |
| `web/3d/` | the 3D page: city, rig, poses, speech bubble (see its README) |
| `web/chat.html` | the older flat chat page, at `/classic` |
| `capri_project_capricorn/` | the 3D model (CC-BY-4.0, see `CREDITS.md`) |

The settings panel has a dose slider: 0 is the stock assistant, 0.74 is the dose she was trained at, and past about 1
she starts to come apart.

## Known problems

- She often won't give her name, and sometimes invents one. A token swap in `server.py` turns "Qwen" into "Capri"
  when the base model's own identity leaks; it hides the problem rather than fixing it.
- Once she's said bye, she tends to keep saying it. A lower dose helps.
- She states made-up facts with full confidence.

## Credits

See `CREDITS.md`. Unofficial fan project, not affiliated with the show's makers.
