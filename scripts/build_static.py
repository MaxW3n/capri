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


NOTICE = """<style>
#demo-note { position: fixed; inset: 0; z-index: 60; display: grid; place-items: center; padding: 16px; background: rgba(7,4,10,.55);
  opacity: 0; visibility: hidden; transition: opacity .35s, visibility .35s; }
#demo-note.on { opacity: 1; visibility: visible; }
#demo-note .chamf { --ch: 16px; --bc: rgba(254,3,83,.7); width: min(460px, 100%); transform: translateY(8px); transition: transform .35s; }
#demo-note.on .chamf { transform: none; }
#demo-note .in { background: var(--panel-solid); clip-path: inherit; padding: 22px 24px 20px; }
#demo-note h2 { margin: 0 0 12px; font: 700 13px/1 var(--label); letter-spacing: .22em; color: var(--pink); }
#demo-note p { margin: 0 0 12px; font: 400 15px/1.5 var(--body); color: var(--white); }
#demo-note .links { display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 18px; }
#demo-note .links a { font: 13px/1 var(--mono); letter-spacing: .06em; color: var(--green); text-decoration: none;
  border: 1px solid rgba(25,242,103,.45); padding: 9px 12px; }
#demo-note .links a:hover { background: rgba(25,242,103,.1); }
#demo-note button { font: 700 12px/1 var(--label); letter-spacing: .2em; color: var(--void); background: var(--pink); border: 0;
  padding: 11px 18px; cursor: pointer; }
#demo-note button:focus-visible, #demo-note a:focus-visible { outline: 2px solid var(--white); outline-offset: 2px; }
</style>
<div id="demo-note" role="dialog" aria-modal="true" aria-labelledby="demo-note-h" hidden>
  <div class="chamf"><div class="in">
    <h2 id="demo-note-h">DEMO ONLY</h2>
    <p>Capri isn't generating anything here. Her replies are canned: the real model's recorded answers to 60 test
      questions, picked by matching the words you type, plus a few stock lines for greetings and questions nothing matches.</p>
    <p>The real thing is a steered Qwen3.8-27B that runs on your own Mac:</p>
    <div class="links">
      <a href="https://huggingface.co/MaxW3n/capri-steering" target="_blank" rel="noopener">HUGGING FACE →</a>
      <a href="https://github.com/MaxW3n/capri" target="_blank" rel="noopener">GITHUB →</a>
    </div>
    <button type="button" id="demo-note-ok">GOT IT</button>
  </div></div>
</div>
<script>
(() => {
  const note = document.getElementById('demo-note'), ok = document.getElementById('demo-note-ok');
  const close = () => { note.classList.remove('on'); setTimeout(() => { note.hidden = true; }, 400); document.getElementById('input')?.focus(); };
  const open = () => { note.hidden = false; requestAnimationFrame(() => note.classList.add('on')); ok.focus(); };
  ok.addEventListener('click', close);
  note.addEventListener('click', (e) => { if (e.target === note) close(); });
  addEventListener('keydown', (e) => { if (e.key === 'Escape' && !note.hidden) close(); });
  const whenBooted = () => document.body.classList.contains('booted') ? setTimeout(open, 900) : setTimeout(whenBooted, 200);
  whenBooted();
})();
</script>
"""

link = f'<a href="{REPO}" target="_blank" rel="noopener"'
edit("index.html", [
    ('"/web/vendor/three/', f'"{PREFIX}/web/vendor/three/'),
    ('src="/web/3d/main.js"', f'src="{PREFIX}/web/3d/main.js"'),
    ('<a class="hlink" href="/classic" title="The original flat chat page">CLASSIC UI</a>',
     f'<a class="hlink" href="{REPO}" target="_blank" rel="noopener" title="Run the real model on your Mac">GET THE MODEL</a>'),
    ('<div class="prow"><span>Flat chat</span><a href="/classic">OPEN CLASSIC UI →</a></div>',
     f'<div class="prow"><span>Real model</span>{link}>GITHUB →</a></div>'),
    ('<a href="/classic">CLASSIC UI →</a>', f'{link}>GET THE MODEL →</a>'),
    ('</body>', NOTICE + '</body>'),
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
