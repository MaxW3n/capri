# Capri

[Try the browser demo](https://maxwen.ca/capri), or run the real model on your own Mac using the setup below.

Chat with Capri, a fan take on the character from Cyberpunk: Edgerunners 2, in a 3D neon city. She's stock
Qwen3.8-27B with no system prompt and no fine-tuned weights. One vector of 5,120 numbers is added after layer 20
on every token to give her a personality. The vector is also on [Hugging Face](https://huggingface.co/MaxW3n/capri-steering).

## Requirements

- An Apple Silicon Mac running macOS **26.2 or later**.
- At least **24 GB of unified memory**, with 32 GB or more recommended. Close other large apps: the model needs
  about 20 GB of available memory.
- **Python 3.13** and an internet connection for setup. The first model download is about **16.1 GB**, plus the
  Python packages. Later starts reuse the saved model.

## Start Capri

1. Install [Python 3.13 for macOS](https://www.python.org/downloads/release/python-31317/). On that page, choose
   **Download macOS installer**, open the downloaded installer and complete it.
2. [Download Capri as a ZIP](https://github.com/MaxW3n/capri/archive/refs/heads/main.zip). Double-click the ZIP in
   Downloads to extract the `capri-main` folder.
3. Open **Terminal** from Applications → Utilities, paste this command, and press Return:

   ```bash
   bash ~/Downloads/capri-main/start.sh
   ```

   The script checks your Mac, installs Capri's packages into its own `.venv` folder, then starts the local model
   server. It does not change your other Python projects.
4. Leave Terminal open. In **Chrome or Edge**, open [maxwen.ca/capri](https://maxwen.ca/capri), click **Connect local
   Capri**, and allow local-network access if asked. Wait for the model download and loading to finish.

If your browser blocks the website's local connection, including in Safari, open
[the local 3D chat](http://127.0.0.1:8777) instead. It uses the same model and interface.

Press **Control-C in Terminal** to stop the server. To start again, run the same command. An interrupted model
download resumes when you restart; installed packages and completed downloads are reused. The model is stored
in Hugging Face's cache, normally `~/.cache/huggingface/hub`.

If you saved the folder somewhere else, type `bash ` in Terminal, drag `start.sh` into the window, then press Return.

Already use Git? After installing Python 3.13:

```bash
git clone https://github.com/MaxW3n/capri.git
cd capri
bash start.sh
```

## How she works

`server.py` loads the model and attaches `capri.safetensors` through `hooks.py`. The 3D scene lives in `web/3d/`;
see [its README](web/3d/README.md). The older flat chat is at `/classic`. For terminal chat, run
`.venv/bin/python capri_chat.py` after setup, with the web server stopped.

The settings panel has a dose slider: 0 is the stock assistant, 0.74 is the trained dose, and past about 1
she starts to come apart.

## Known problems

- She often won't give her name, and sometimes invents one. A token swap in `server.py` turns "Qwen" into "Capri"
  when the base model's identity leaks; it hides the problem rather than fixing it.
- Once she's said bye, she tends to keep saying it. A lower dose helps.
- She states made-up facts with full confidence.

## Credits

See [CREDITS.md](CREDITS.md) for the CC-BY-4.0 character model and other credits. Unofficial fan project,
not affiliated with the show's makers.
