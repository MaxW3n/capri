"""Real loopback HTTP tests. No MLX, model downloads, browsers or GPU work."""
import http.client
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_http import LocalServer, ModelState, PROTOCOL
import server as model_server


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="capri-http-test-")
        self.root = Path(self.temp.name)
        (self.root / "web/3d").mkdir(parents=True)
        (self.root / "web/3d/index.html").write_text("<title>Capri test frontend</title>")
        (self.root / "private.txt").write_text("must not be served")
        (self.root / "web/3d/leak.txt").symlink_to(self.root / "private.txt")
        self.state = ModelState()
        self.chats = []

        def chat(payload):
            self.chats.append(payload)
            output = queue.Queue()
            for event in ({"t": "Test reply."}, {"done": True, "tps": 1}, None):
                output.put(event)
            return output, threading.Event()

        self.http = LocalServer(self.state, chat, root=self.root, port=0)
        self.thread = threading.Thread(target=lambda: self.http.serve_forever(poll_interval=0.02), daemon=True)
        self.thread.start()

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method="GET", path="/info", *, origin="https://maxwen.ca", host="127.0.0.1:8777", body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=2)
        conn.putrequest(method, path, skip_host=True)
        if host is not None:
            conn.putheader("Host", host)
        if origin is not None:
            conn.putheader("Origin", origin)
        if body is not None:
            conn.putheader("Content-Length", str(len(body.encode("utf-8"))))
        for name, value in (headers or []):
            conn.putheader(name, value)
        conn.endheaders(body.encode("utf-8") if body is not None else None)
        response = conn.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        conn.close()
        return result

    def test_listener_is_ipv4_loopback_only(self):
        self.assertEqual(self.http.server_address[0], "127.0.0.1")

    def test_loading_info_has_protocol_and_allowed_cors(self):
        code, headers, body = self.request()
        info = json.loads(body)
        self.assertEqual(code, 200)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://maxwen.ca")
        self.assertEqual(headers["Vary"], "Origin")
        self.assertEqual(info["protocol"], PROTOCOL)
        self.assertEqual(info["status"], "loading")
        self.assertFalse(info["ready"])
        self.assertEqual((info["size"], info["default_dose"]), ("27B", 0.74))

    def test_local_origins_and_www_are_explicitly_allowed(self):
        for origin in ("https://www.maxwen.ca", "http://127.0.0.1:8777", "http://localhost:8777"):
            with self.subTest(origin=origin):
                code, headers, _ = self.request(origin=origin)
                self.assertEqual(code, 200)
                self.assertEqual(headers["Access-Control-Allow-Origin"], origin)

    def test_untrusted_and_null_origins_cannot_read_status(self):
        for origin in ("https://evil.example", "https://maxwen.ca.evil.example", "null"):
            with self.subTest(origin=origin):
                code, headers, _ = self.request(origin=origin)
                self.assertEqual(code, 403)
                self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_missing_foreign_and_duplicate_hosts_are_denied(self):
        for host in (None, "evil.example:8777", "127.0.0.1:9999"):
            with self.subTest(host=host):
                code, headers, _ = self.request(host=host)
                self.assertEqual(code, 403)
                self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(self.request(headers=[("Host", "localhost:8777")])[0], 403)

    def test_command_line_json_requests_need_no_origin(self):
        code, headers, _ = self.request(origin=None)
        self.assertEqual(code, 200)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_preflight_allows_only_standard_methods_and_content_type(self):
        code, headers, body = self.request("OPTIONS", "/chat", headers=[
            ("Access-Control-Request-Method", "POST"),
            ("Access-Control-Request-Headers", "content-type"),
            ("Access-Control-Request-Private-Network", "true"),
        ])
        self.assertEqual((code, body), (204, b""))
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://maxwen.ca")
        self.assertEqual(headers["Access-Control-Allow-Methods"], "GET, POST")
        self.assertEqual(headers["Access-Control-Allow-Headers"], "Content-Type")
        self.assertNotIn("Access-Control-Allow-Private-Network", headers)
        for header in (("Access-Control-Request-Method", "DELETE"), ("Access-Control-Request-Headers", "Authorization")):
            with self.subTest(header=header):
                self.assertEqual(self.request("OPTIONS", "/chat", headers=[header])[0], 403)

    def test_untrusted_preflight_has_no_cors_grant(self):
        code, headers, _ = self.request("OPTIONS", "/chat", origin="https://evil.example")
        self.assertEqual(code, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_chat_is_unavailable_during_loading_and_after_load_error(self):
        for status in ("loading", "error"):
            self.state.set(status, "Model is unavailable")
            code, headers, body = self.request("POST", "/chat", body='{"messages": []}', headers=[("Content-Type", "application/json")])
            self.assertEqual(code, 503)
            self.assertEqual(json.loads(body)["status"], status)
            self.assertEqual(headers["Access-Control-Allow-Origin"], "https://maxwen.ca")
        self.assertEqual(self.chats, [])

    def test_json_content_type_is_exact_not_a_prefix(self):
        for content_type in ("text/plain", "application/json-malicious", "application/x-www-form-urlencoded"):
            with self.subTest(content_type=content_type):
                self.assertEqual(self.request("POST", "/chat", body="{}", headers=[("Content-Type", content_type)])[0], 403)

    def test_ready_chat_stream_and_errors_have_cors(self):
        self.state.set("ready", "Ready")
        code, headers, body = self.request("POST", "/chat", body='{"messages": [{"role":"user","content":"hi"}]}', headers=[("Content-Type", "application/json; charset=utf-8")])
        self.assertEqual(code, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://maxwen.ca")
        self.assertIn(b'data: {"t": "Test reply."}\n\n', body)
        self.assertIn(b'"done": true', body)
        code, headers, _ = self.request("POST", "/chat", body="not-json", headers=[("Content-Type", "application/json")])
        self.assertEqual(code, 400)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://maxwen.ca")

    def test_status_reports_load_failure_without_losing_http(self):
        self.state.set("error", "Model download failed")
        code, headers, body = self.request(path="/status")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["message"], "Model download failed")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://maxwen.ca")
        self.assertEqual(self.request(path="/")[0], 200)

    def test_private_paths_and_symlinks_are_not_served(self):
        for path in ("/server.py", "/private.txt", "/web/3d/../../private.txt", "/web/3d/leak.txt"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path=path)[0], 404)


class StartupTests(unittest.TestCase):
    def test_server_import_is_independent_of_model_packages(self):
        source = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import server; assert not any(n in sys.modules for n in ('mlx', 'mlx_lm', 'numpy', 'safetensors', 'huggingface_hub', 'hooks'))"
        result = subprocess.run([sys.executable, "-I", "-c", source], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_port_collision_never_initialises_a_model(self):
        initializer = Mock()
        with patch.object(model_server, "LocalServer", side_effect=OSError("already in use")):
            with self.assertRaises(OSError):
                model_server.run(initializer=initializer)
        initializer.assert_not_called()

    def test_http_is_available_while_initializer_waits_and_after_it_fails(self):
        published = queue.Queue()
        entered, release, stop = threading.Event(), threading.Event(), threading.Event()

        def initializer(state):
            entered.set()
            release.wait(timeout=3)
            raise RuntimeError("test download failed")

        thread = threading.Thread(target=lambda: model_server.run(initializer=initializer,
                                  stop_event=stop, port=0, on_listening=published.put), daemon=True)
        thread.start()
        listener = published.get(timeout=2)

        def get_info():
            conn = http.client.HTTPConnection("127.0.0.1", listener.server_port, timeout=2)
            conn.request("GET", "/info", headers={"Host": "127.0.0.1:8777", "Origin": "https://maxwen.ca"})
            response = conn.getresponse()
            result = response.status, json.loads(response.read())
            conn.close()
            return result

        try:
            self.assertTrue(entered.wait(timeout=2))
            self.assertEqual(get_info()[1]["status"], "loading")
            release.set()
            deadline = time.monotonic() + 2
            info = get_info()[1]
            while info["status"] != "error" and time.monotonic() < deadline:
                time.sleep(0.01)
                info = get_info()[1]
            self.assertEqual(info["status"], "error")
            self.assertIn("test download failed", info["message"])
            self.assertFalse(info["ready"])
        finally:
            release.set()
            stop.set()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
