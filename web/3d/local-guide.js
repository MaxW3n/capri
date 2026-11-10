// Hosted demo's optional setup panel. No loopback fetch occurs in this module.
const byId = (id) => document.getElementById(id);
const guide = byId('local-guide'), intro = byId('local-intro');
const connectButton = byId('local-connect');
const introVersion = '2';
let returnFocus = null, chat = null;
function openGuide() {
  if (byId('drawer')?.getAttribute('aria-hidden') === 'false') byId('btn-close')?.click();
  if (byId('btn-set')?.getAttribute('aria-expanded') === 'true') byId('btn-set')?.click();
  if (intro.open) intro.close();
  returnFocus = document.activeElement?.closest('#local-intro') ? byId('btn-run-local') : document.activeElement;
  if (!guide.open) guide.showModal();
  byId('local-close').focus();
}
function closeGuide() { guide.close(); returnFocus?.focus(); }
function closeIntro() {
  intro.close();
  try { localStorage.setItem('capri.web-intro-seen', introVersion); } catch { /* private browser */ }
}
function render(state) {
  const names = { idle: 'Demo mode', connecting: 'Requesting local connection', loading: 'Capri is loading',
    ready: 'Local Capri connected', denied: 'Browser permission blocked', lost: 'Local connection lost', error: 'Cannot connect yet' };
  byId('local-feedback').dataset.state = state.status;
  byId('local-feedback-title').textContent = names[state.status] || 'Local connection';
  byId('local-feedback-message').textContent = state.message;
  const waiting = state.status === 'connecting' || state.status === 'loading';
  connectButton.disabled = waiting || state.status === 'ready' || !chat;
  connectButton.textContent = state.status === 'ready' ? 'Connected' : waiting ? 'Waiting for Capri…'
    : ['error', 'denied', 'lost'].includes(state.status) ? 'Try connecting again' : 'Connect local Capri';
  byId('local-demo').textContent = state.status === 'idle' ? 'Back to demo' : 'Disconnect · use demo';
  byId('local-remedy').hidden = !['error', 'denied', 'lost'].includes(state.status);
}
function attach(client) { chat = client; render(chat.connection); }
for (const button of document.querySelectorAll('[data-open-local], #btn-run-local')) button.addEventListener('click', openGuide);
byId('local-close').addEventListener('click', closeGuide);
byId('intro-demo').addEventListener('click', closeIntro);
byId('intro-setup').addEventListener('click', () => { closeIntro(); openGuide(); });
byId('local-demo').addEventListener('click', () => { chat?.disconnect(); closeGuide(); });
connectButton.addEventListener('click', () => {
  if (!chat) return;
  // This explicit gesture is the first operation that can request loopback access.
  chat.connect();
});
byId('local-copy').addEventListener('click', async () => {
  const button = byId('local-copy');
  try {
    await navigator.clipboard.writeText(byId('local-command').textContent);
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = 'Copy'; }, 1800);
  } catch {
    const selection = window.getSelection(), range = document.createRange();
    range.selectNodeContents(byId('local-command'));
    selection.removeAllRanges(); selection.addRange(range);
    button.textContent = 'Select + copy';
  }
});
window.addEventListener('capri:connection', (event) => render(event.detail));
window.addEventListener('capri:ready', (event) => attach(event.detail.chat));
if (window.__capri?.chat) attach(window.__capri.chat);
else connectButton.disabled = true;
for (const dialog of [intro, guide]) {
  dialog.addEventListener('click', (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) {
      if (dialog === intro) closeIntro(); else closeGuide();
    }
  });
}
function welcome() {
  let seen = false;
  try { seen = localStorage.getItem('capri.web-intro-seen') === introVersion; } catch { /* private browser */ }
  if (seen || intro.open || guide.open) return;
  intro.showModal();
}
const whenBooted = () => document.body.classList.contains('booted') ? setTimeout(welcome, 700) : setTimeout(whenBooted, 200);
whenBooted();
