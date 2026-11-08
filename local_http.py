"""Loopback-only HTTP transport. Uses the standard library, never imports a model.

The hosted Capri page must ask the visitor before connecting. This server permits
only Capri's published website and its own local frontend; CORS never uses '*'.
"""
import json
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit


PORT = 8777
PROTOCOL = "capri-local-v1"
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
ALLOWED_ORIGINS = {
    "https://maxwen.ca", "https://www.maxwen.ca",
    f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}",
}
API_PATHS = {"/info", "/status", "/chat"}
STATIC_ROOTS = ("web/3d/", "web/vendor/three/", "capri_project_capricorn/")
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json", ".gltf": "model/gltf+json",
    ".bin": "application/octet-stream", ".png": "image/png", ".webp": "image/webp",
    ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8",
    ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf",
}
MAX_BODY_BYTES = 256 * 1024


class ModelState:
    """One locked status snapshot, readable while the main thread loads MLX."""

    def __init__(self, *, layer=20, size="27B", min_dose=0.0, max_dose=1.2, default_dose=0.74):
        self._lock = threading.Lock()
        self._info = {"protocol": PROTOCOL, "status": "loading", "ready": False,
                      "message": "Starting Capri's model runner…", "layer": layer, "size": size,
                      "min_dose": min_dose, "max_dose": max_dose, "default_dose": default_dose}

    def set(self, status, message):
        if status not in {"loading", "ready", "error"}:
            raise ValueError("invalid model status")
        with self._lock:
            self._info.update(status=status, ready=status == "ready", message=str(message))

    def snapshot(self):
        with self._lock:
            return dict(self._info)


class LocalHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _host_ok(self):
        hosts = self.headers.get_all("Host", [])
        return len(hosts) == 1 and hosts[0] in ALLOWED_HOSTS

    def _origin(self):
        origins = self.headers.get_all("Origin", [])
        return origins[0] if len(origins) == 1 else None

    def _authorise(self):
        origins = self.headers.get_all("Origin", [])
        if not self._host_ok() or len(origins) > 1 or (origins and origins[0] not in ALLOWED_ORIGINS):
            self._json(403, {"error": "This host or website is not allowed to access Capri."})
            return False
        return True

    def _cors(self):
        self.send_header("Vary", "Origin")
        origin = self._origin()
        if self._host_ok() and origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, payload):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _path(self):
        return unquote(urlsplit(self.path).path)

    def do_OPTIONS(self):
        if not self._authorise():
            return
        if self._path() not in API_PATHS:
            return self._json(404, {"error": "not found"})
        method = self.headers.get("Access-Control-Request-Method", "GET").upper()
        headers = {h.strip().lower() for h in self.headers.get("Access-Control-Request-Headers", "").split(",") if h.strip()}
        if method not in {"GET", "POST"} or not headers.issubset({"content-type"}):
            return self._json(403, {"error": "This request method or header is not allowed."})
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self._cors()
        self.end_headers()

    def _file(self, path):
        if not path.is_file():
            return False
        self._send(200, path.read_bytes(), STATIC_TYPES.get(path.suffix.lower(), "application/octet-stream"))
        return True

    def _static(self, path):
        rel = os.path.normpath(path.lstrip("/"))
        for prefix in STATIC_ROOTS:
            if not rel.startswith(prefix):
                continue
            directory = (self.server.root / prefix).resolve()
            full = (self.server.root / rel).resolve()
            if directory.is_relative_to(self.server.root) and full.is_relative_to(directory):
                return self._file(full)
        return False

    def do_GET(self):
        if not self._authorise():
            return
        path = self._path()
        if path in {"/info", "/status"}:
            return self._json(200, self.server.state.snapshot())
        if path in {"/", "/index.html", "/3d"}:
            if self._file(self.server.root / "web/3d/index.html"):
                return
        elif path in {"/classic", "/classic.html"}:
            source = self.server.root / "web/chat.html"
            if source.is_file():
                info = self.server.state.snapshot()
                html = source.read_text().replace('max="1.2"', f'max="{info["max_dose"]}"') \
                    .replace('value="0.85"', f'value="{info["default_dose"]}"').replace('>0.85<', f'>{info["default_dose"]:.2f}<') \
                    .replace('0.85 Capri', f'{info["default_dose"]:.2f} Capri').replace('steered 4B', f'steered {info["size"]}') \
                    .replace('at layer 10', f'at layer {info["layer"]}')
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        elif self._static(path):
            return
        self._json(404, {"error": "not found"})

    def _body(self):
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
            raise ValueError("one Content-Length header is required")
        length = int(lengths[0])
        if not 0 < length <= MAX_BODY_BYTES:
            raise ValueError(f"request body must be between 1 and {MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("request body ended early")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")
        return payload

    def do_POST(self):
        if not self._authorise():
            return
        if self._path() != "/chat":
            return self._json(404, {"error": "not found"})
        content_types = self.headers.get_all("Content-Type", [])
        if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != "application/json":
            return self._json(403, {"error": "Content-Type must be application/json."})
        info = self.server.state.snapshot()
        if not info["ready"]:
            return self._json(503, {"error": info["message"], "protocol": PROTOCOL, "status": info["status"]})
        try:
            payload = self._body()
            output, stop = self.server.chat(payload)
        except (ValueError, TypeError, KeyError) as error:
            return self._json(400, {"error": f"bad request: {error}"})
        except Exception:
            return self._json(500, {"error": "Capri could not prepare this reply. Check the model runner log."})

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            while True:
                event = output.get()
                if event is None:
                    break
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            stop.set()


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state, chat, *, root, port=PORT):
        # The port override is for isolated CPU tests; the public entry point
        # always uses PORT. Binding a public interface is intentionally absent.
        super().__init__(("127.0.0.1", port), LocalHandler)
        self.state = state
        self.chat = chat
        self.root = Path(root).resolve()

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)
