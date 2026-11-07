"""Build the static demo of the 3D page for a website subfolder (maxwen.ca/capri).

    python scripts/build_static.py ~/Code/maxwen.ca/public/capri

Copies web/3d, three.js and the 3D model into OUT, then rewrites the copy (the sources stay as they are):
  - every root path (/web/..., /capri_project_capricorn/...) gets the PREFIX, since the page no longer sits at /
  - the page never looks for server.py: replies always come from the recorded demo lines
  - "start server.py" hints become a link to the GitHub repo
"""
import shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]).expanduser().resolve()
PREFIX = "/capri"
REPO = "https://github.com/MaxW3n/capri"

if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(ROOT / "web/3d", OUT / "web/3d", ignore=shutil.ignore_patterns("README.md", "demo.html"))
shutil.copytree(ROOT / "web/vendor/three", OUT / "web/vendor/three")
shutil.copytree(ROOT / "capri_project_capricorn", OUT / "capri_project_capricorn")
shutil.move(OUT / "web/3d/index.html", OUT / "index.html")


def edit(path, pairs):
    p = OUT / path
    s = p.read_text()
    for old, new in pairs:
        if old not in s:
            sys.exit(f"{path}: not found: {old[:70]!r}")
        s = s.replace(old, new)
    p.write_text(s)


link = f'<a href="{REPO}" target="_blank" rel="noopener"'
edit("index.html", [
    ('"/web/vendor/three/', f'"{PREFIX}/web/vendor/three/'),
    ('src="/web/3d/main.js"', f'src="{PREFIX}/web/3d/main.js"'),
    ('<a class="hlink" href="/classic" title="The original flat chat page">CLASSIC UI</a>',
     f'<a class="hlink" href="{REPO}" target="_blank" rel="noopener" title="Run the real model on your Mac">GET THE MODEL</a>'),
    ('<div class="prow"><span>Flat chat</span><a href="/classic">OPEN CLASSIC UI →</a></div>',
     f'<div class="prow"><span>Real model</span>{link}>GITHUB →</a></div>'),
    ('<a href="/classic">CLASSIC UI →</a>', f'{link}>GET THE MODEL →</a>'),
    ('DEMO: replies are recordings made at 1.00. Start server.py to steer her live.',
     f'DEMO: replies are recordings from the real model, so the slider does nothing here. '
     f'To steer her live, run her on your Mac: {link}>github.com/MaxW3n/capri</a>'),
])
edit("web/3d/main.js", [
    ("'/web/3d/bubble.js'", f"'{PREFIX}/web/3d/bubble.js'"),
    ("'/web/3d/city.js'", f"'{PREFIX}/web/3d/city.js'"),
    ("'/web/3d/capri.js'", f"'{PREFIX}/web/3d/capri.js'"),
    ("WebGL is not available in this browser. Try /classic.", "WebGL is not available in this browser."),
    ("url: '/capri_project_capricorn/scene.gltf'", f"url: '{PREFIX}/capri_project_capricorn/scene.gltf'"),
    ("'<span class=\"long\">DEMO MODE — start server.py for the real Capri</span><span class=\"short\">DEMO MODE</span>'",
     f"'<span class=\"long\">DEMO · RECORDED REPLIES · {link} style=\"color:inherit;pointer-events:auto\">GET THE REAL MODEL →</a></span><span class=\"short\">DEMO</span>'"),
    ("'Demo mode: replies come from recorded lines. Start server.py for the real Capri.'",
     f"'Demo: replies are recordings from the real model. Run her live on your Mac: {REPO}'"),
])
edit("web/3d/capri.js", [
    ("opts.url || '/capri_project_capricorn/scene.gltf'", f"opts.url || '{PREFIX}/capri_project_capricorn/scene.gltf'"),
])
edit("web/3d/chat.js", [
    ("fetch('/web/3d/demo_lines.json')", f"fetch('{PREFIX}/web/3d/demo_lines.json')"),
    ("    lastProbe = Date.now();\n", "    lastProbe = Date.now();\n    return false; // static demo: there is no model server here\n"),
    ("Ask the real me, she's smarter. Start server.py.", "Ask the real me, she's smarter. She's on GitHub."),
    ("...Link's dead. Start server.py and try me again.", "...Link's dead. Try me again."),
])
print(f"built {OUT}")
