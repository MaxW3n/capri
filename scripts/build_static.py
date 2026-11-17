"""Build Capri's recorded demo and optional live browser or Terminal chat at a website subfolder.

    python scripts/build_static.py ~/Code/maxwen.ca/public/capri

Only the generated copy gets /capri paths and the manual-connect setup guide.
Source assets still work at the root of the local Python server. The website
starts in recorded-demo mode and never probes localhost before a Connect click.
"""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]).expanduser().resolve()
PREFIX = "/capri"
if OUT == ROOT or ROOT.is_relative_to(OUT) or OUT.name != "capri":
    raise SystemExit("Output must be a website's generated capri subfolder, outside the source repository.")
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(ROOT / "web/3d", OUT / "web/3d", ignore=shutil.ignore_patterns("README.md", "demo.html"))
shutil.copytree(ROOT / "web/vendor/three", OUT / "web/vendor/three")
shutil.copytree(ROOT / "web/browser", OUT / "browser")
shutil.copytree(ROOT / "capri_project_capricorn", OUT / "capri_project_capricorn")
shutil.move(OUT / "web/3d/index.html", OUT / "index.html")


def edit(path, replacements):
    file = OUT / path
    text = file.read_text()
    for old, new in replacements:
        if old not in text:
            raise SystemExit(f"{path}: missing expected source fragment: {old[:90]!r}")
        text = text.replace(old, new)
    file.write_text(text)


config = json.dumps({
    "manualLocal": True,
    "apiBase": "http://127.0.0.1:8777",
    "browserWorkerUrl": f"{PREFIX}/browser/v1/inference-v2.worker.js",
})
guide = (ROOT / "web/3d/local-guide.html").read_text()
edit("index.html", [
    ('<script type="importmap">', f'<script>window.CAPRI_CONFIG = {config};</script>\n<script type="importmap">'),
    ('"/web/vendor/three/', f'"{PREFIX}/web/vendor/three/'),
    ('src="/web/3d/main.js"', f'src="{PREFIX}/web/3d/main-v3.js"'),
    ('</head>', f'<link rel="stylesheet" href="{PREFIX}/web/3d/local-guide-v5.css">\n</head>'),
    ('<a class="hlink" href="/classic" title="The original flat chat page">CLASSIC UI</a>',
     '<button class="local-primary" type="button" id="btn-browser-start" disabled>Start</button>\n  '
     '<button class="hlink" type="button" id="btn-run-local" aria-haspopup="dialog" aria-controls="local-guide">RUN LOCALLY</button>'),
    ('<div class="prow"><span>Flat chat</span><a href="/classic">OPEN CLASSIC UI →</a></div>',
     '<div class="prow"><span>Local setup</span><button class="local-settings-link" type="button" data-open-local>RUN LOCALLY →</button></div>'),
    ('<a href="/classic">CLASSIC UI →</a>', '<button class="local-settings-link" type="button" data-open-local>RUN LOCALLY →</button>'),
    ('<span class="long">CONNECTING…</span><span class="short">LINK…</span>',
     '<span class="long">DEMO · RECORDED REPLIES</span><span class="short">DEMO</span>'),
    ('DEMO: replies are recordings made at 1.00. Start server.py to steer her live.',
     'Demo replies are recordings. The intensity slider affects only live replies.'),
    ('</body>', guide + f'\n<script type="module" src="{PREFIX}/web/3d/local-guide-v4.js"></script>\n</body>'),
])
edit("web/3d/main.js", [
    ("'./chat.js'", "'./scene-chat-v1.js'"),
    ("'/web/3d/bubble.js'", f"'{PREFIX}/web/3d/bubble.js'"),
    ("'/web/3d/city.js'", f"'{PREFIX}/web/3d/city.js'"),
    ("'/web/3d/capri.js'", f"'{PREFIX}/web/3d/capri.js'"),
    ("WebGL is not available in this browser. Try /classic.", "WebGL is not available in this browser."),
    ("url: '/capri_project_capricorn/scene.gltf'", f"url: '{PREFIX}/capri_project_capricorn/scene.gltf'"),
])
edit("web/3d/capri.js", [
    ("opts.url || '/capri_project_capricorn/scene.gltf'", f"opts.url || '{PREFIX}/capri_project_capricorn/scene.gltf'"),
])
edit("web/3d/chat.js", [
    ("fetch('/web/3d/demo_lines.json')", f"fetch('{PREFIX}/web/3d/demo_lines.json')"),
    ("Ask the real me, she's smarter. Start server.py.", "Ask the real me, she's smarter. Run me locally."),
    ("...Link's dead. Start server.py and try me again.", "...Link's dead. Try me again."),
])
edit("web/3d/scene-chat.js", [
    ("'./chat.js'", "'./chat-v3.js'"),
])
# The production CDN may keep old responses at previously published URLs.
for src, dest in [
    ("web/3d/main.js", "web/3d/main-v3.js"),
    ("web/3d/scene-chat.js", "web/3d/scene-chat-v1.js"),
    ("web/3d/chat.js", "web/3d/chat-v3.js"),
    ("web/3d/local-guide.css", "web/3d/local-guide-v5.css"),
    ("web/3d/local-guide.js", "web/3d/local-guide-v4.js"),
    ("browser/v1/inference.worker.js", "browser/v1/inference-v2.worker.js"),
]:
    (OUT / src).rename(OUT / dest)
print(f"Built recorded demo, browser model and manual local setup: {OUT}")
