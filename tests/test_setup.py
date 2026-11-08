"""Exercise setup with fake system/Python commands; never install or load a model."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


SOURCE = Path(__file__).resolve().parents[1]


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="capri setup test ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.repo = self.root / "Capri with spaces"
        self.repo.mkdir()
        for name in ("start.sh", "requirements.txt"):
            shutil.copy2(SOURCE / name, self.repo / name)
        (self.repo / "server.py").write_text("raise AssertionError('The real server must not run in setup tests')\n")
        self.bin = self.root / "fake-bin"
        self.bin.mkdir()
        self.log = self.root / "commands.jsonl"
        self.executable("uname", "#!/bin/sh\nif [ \"$1\" = -s ]; then echo \"${FAKE_OS:-Darwin}\"; else echo \"${FAKE_ARCH:-arm64}\"; fi\n")
        self.executable("sw_vers", "#!/bin/sh\necho \"${FAKE_MACOS:-26.2}\"\n")
        self.executable("sysctl", "#!/bin/sh\necho \"${FAKE_MEMORY:-34359738368}\"\n")
        fake_python = "#!" + sys.executable + "\n" + textwrap.dedent('''\
            import hashlib
            import json
            import os
            from pathlib import Path
            import sys

            args = sys.argv[1:]
            root = Path.cwd()
            installed = root / ".venv" / ".fake-installed"
            def record(kind):
                with open(os.environ["FAKE_LOG"], "a") as log:
                    log.write(json.dumps({"kind": kind, "args": args, "cwd": str(root), "python": sys.argv[0]}) + "\\n")
            if args[:2] == ["-m", "venv"]:
                record("venv")
                target = Path(args[2]) / "bin" / "python"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(Path(__file__).resolve())
            elif args[:2] == ["-m", "pip"]:
                record("install" if "install" in args else "pip-check")
                if "install" in args:
                    if os.environ.get("FAKE_INSTALL_FAIL") == "1": sys.exit(12)
                    installed.touch()
            elif args[:1] == ["-c"]:
                code = args[1]
                if "hashlib.sha256" in code:
                    print(hashlib.sha256(Path(args[2]).read_bytes()).hexdigest())
                elif "sys.version_info" in code:
                    if os.environ.get("FAKE_PYTHON_BAD") == "1": sys.exit(1)
                    if ".venv" in Path(sys.argv[0]).parts and os.environ.get("FAKE_VENV_BAD") == "1": sys.exit(1)
                elif "sys.prefix" in code:
                    if os.environ.get("FAKE_NOT_VENV") == "1": sys.exit(1)
                else:
                    raise AssertionError("Unexpected setup Python expression")
            elif args[:1] == ["-"]:
                code = sys.stdin.read()
                assert "importlib.import_module" in code
                record("imports")
                if not installed.exists() or os.environ.get("FAKE_IMPORT_FAIL") == "1": sys.exit(1)
            elif args == ["-u", "server.py"]:
                record("server")
            else:
                raise AssertionError("Unexpected Python call: " + repr(args))
            ''')
        self.python = self.executable("python3.13", fake_python)
        self.environment = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.defpath,
                                CAPRI_PYTHON=str(self.python), FAKE_LOG=str(self.log))

    def executable(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)
        return path

    def run_setup(self, *args, **overrides):
        return subprocess.run(["/bin/bash", str(self.repo / "start.sh"), *args], cwd=self.root,
                              env=dict(self.environment, **overrides), text=True, capture_output=True, timeout=15)

    def events(self, kind=None):
        events = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return [event for event in events if kind is None or event["kind"] == kind]

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_help_needs_neither_supported_mac_nor_python(self):
        result = self.run_setup("--help", FAKE_OS="Linux", FAKE_PYTHON_BAD="1")
        self.assert_success(result)
        self.assertIn("--check", result.stdout)
        self.assertFalse((self.repo / ".venv").exists())
        self.assertEqual(self.events(), [])

    def test_check_does_not_install_or_start_a_model(self):
        result = self.run_setup("--check")
        self.assert_success(result)
        self.assertIn("Checks passed", result.stdout)
        self.assertFalse((self.repo / ".venv").exists())
        self.assertEqual(self.events(), [])

    def test_unsupported_mac_or_python_stops_before_mutating(self):
        for override, message in (
            ({"FAKE_OS": "Linux"}, "macOS"),
            ({"FAKE_ARCH": "x86_64"}, "Rosetta"),
            ({"FAKE_MACOS": "26.1"}, "26.2"),
            ({"FAKE_MEMORY": str(16 * 1024 ** 3)}, "24 GB"),
            ({"FAKE_PYTHON_BAD": "1"}, "Python 3.13"),
        ):
            with self.subTest(override=override):
                result = self.run_setup(**override)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse((self.repo / ".venv").exists())
                self.assertEqual(self.events(), [])

    def test_first_run_uses_private_environment_and_own_directory(self):
        result = self.run_setup()
        self.assert_success(result)
        self.assertEqual(len(self.events("venv")), 1)
        self.assertEqual(len(self.events("install")), 1)
        self.assertEqual(len(self.events("pip-check")), 1)
        self.assertEqual(len(self.events("server")), 1)
        for event in self.events():
            self.assertEqual(Path(event["cwd"]), self.repo)
            if event["kind"] != "venv":
                self.assertEqual(Path(event["python"]), self.repo / ".venv/bin/python")
        self.assertIn("--isolated", self.events("install")[0]["args"])
        self.assertIn("--only-binary=:all:", self.events("install")[0]["args"])
        self.assertIn("Connect local Capri", result.stdout)
        self.assertIn("http://127.0.0.1:8777", result.stdout)
        stamp = (self.repo / ".venv/.capri-requirements.sha256").read_text().strip()
        self.assertEqual(stamp, hashlib.sha256((self.repo / "requirements.txt").read_bytes()).hexdigest())

    def test_second_run_reuses_packages_and_changed_requirements_reinstall(self):
        self.assert_success(self.run_setup())
        self.assert_success(self.run_setup())
        self.assertEqual(len(self.events("install")), 1)
        self.assertEqual(len(self.events("server")), 2)
        with (self.repo / "requirements.txt").open("a") as requirements:
            requirements.write("# Updated configuration\n")
        self.assert_success(self.run_setup())
        self.assertEqual(len(self.events("install")), 2)
        self.assertEqual(len(self.events("venv")), 1)

    def test_install_failure_can_be_retried_without_a_success_stamp(self):
        failed = self.run_setup(FAKE_INSTALL_FAIL="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("same command again", failed.stderr)
        self.assertFalse((self.repo / ".venv/.capri-requirements.sha256").exists())
        self.assertEqual(self.events("server"), [])
        self.assert_success(self.run_setup())
        self.assertEqual(len(self.events("venv")), 1)
        self.assertEqual(len(self.events("install")), 2)

    def test_incompatible_existing_environment_is_not_deleted_or_used(self):
        self.assert_success(self.run_setup())
        count = len(self.events("server"))
        result = self.run_setup(FAKE_VENV_BAD="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete or incompatible .venv", result.stderr)
        self.assertTrue((self.repo / ".venv/.fake-installed").exists())
        self.assertEqual(len(self.events("server")), count)

    def test_broken_imports_prevent_starting_the_server(self):
        result = self.run_setup(FAKE_IMPORT_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("installed packages could not load", result.stderr)
        self.assertEqual(self.events("server"), [])
        self.assertFalse((self.repo / ".venv/.capri-requirements.sha256").exists())


if __name__ == "__main__":
    unittest.main()
