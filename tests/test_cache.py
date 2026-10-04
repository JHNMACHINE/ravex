"""The cache of compiled wheels and kernels (GPU-181).

What these tests hold it to is the issue's "done when": a second machine with
the same key skips the compilation, and a machine with a different key loads
nothing of the first's. The store is a directory; the GPUs are given rather
than asked of a driver, so each test says which machine it is.
"""

import json
import os

import pytest

from ravex import _cache
from ravex._cli import main


def _fill(directory, files):
    for name, data in files.items():
        path = os.path.join(directory, *name.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)


def _read(directory):
    return {name: open(os.path.join(directory, *name.split("/")), "rb").read() for name in _cache._local_files(directory)}


def test_a_second_machine_with_the_same_key_gets_what_the_first_built(tmp_path):
    store = _cache.LocalStore(str(tmp_path / "store"))
    key = _cache.compute_key("triton", gpus=["sm_89"])
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    _fill(first, {"ab/kernel.cubin": b"\x7fELF", "ab/kernel.json": b"{}"})

    sent = _cache.push(store, key, first)
    got = _cache.pull(store, _cache.compute_key("triton", gpus=["sm_89"]), second)

    assert (sent.hit, sent.files) == (False, 2)
    assert (got.hit, got.files) == (True, 2)
    assert _read(second) == _read(first)


def test_another_gpu_loads_nothing(tmp_path):
    store = _cache.LocalStore(str(tmp_path / "store"))
    first = str(tmp_path / "first")
    _fill(first, {"kernel.cubin": b"built for sm_89"})
    _cache.push(store, _cache.compute_key("triton", gpus=["sm_89"]), first)

    target = str(tmp_path / "other")
    got = _cache.pull(store, _cache.compute_key("triton", gpus=["sm_90"]), target)

    assert not got.hit
    assert _read(target) == {}


def test_a_stored_key_that_differs_is_a_miss_whatever_its_id_says(tmp_path):
    """The id only finds the key: a stored key that does not match field by
    field - a hash collision, or a key someone edited - loads nothing."""
    store = _cache.LocalStore(str(tmp_path / "store"))
    key = _cache.compute_key("wheels", gpus=["sm_89"])
    base = "wheels/%s" % key.id
    forged = dict(key.fields, torch="2.0.0+cu118")
    store.put(base + "/key.json", json.dumps(forged).encode())
    store.put(base + "/files/flash_attn.whl", b"for another torch")

    got = _cache.pull(store, key, str(tmp_path / "wheels"))

    assert not got.hit
    assert "torch" in got.reason
    with pytest.raises(ValueError):
        _cache.push(store, key, str(tmp_path / "wheels"))


def test_a_push_sends_only_what_the_store_lacks(tmp_path):
    store = _cache.LocalStore(str(tmp_path / "store"))
    key = _cache.compute_key("inductor", gpus=[])
    directory = str(tmp_path / "inductor")
    _fill(directory, {"a.py": b"a"})
    _cache.push(store, key, directory)
    _fill(directory, {"b.py": b"bb"})

    again = _cache.push(store, key, directory)

    assert (again.hit, again.files, again.bytes) == (True, 1, 2)


class DictStore:
    """A store that lists whatever names it was given, as a bucket can."""

    def __init__(self):
        self.objects = {}

    def get(self, path):
        return self.objects.get(path)

    def put(self, path, data):
        self.objects[path] = data

    def list(self, prefix):
        return sorted(path for path in self.objects if path.startswith(prefix))


@pytest.mark.parametrize("name", ["../../escape.whl", "/etc/escape.whl", "a//b.whl", "a\\..\\..\\escape.whl"])
def test_a_name_that_leaves_the_directory_is_refused(tmp_path, name):
    store = DictStore()
    key = _cache.compute_key("wheels", gpus=[])
    store.put("wheels/%s/key.json" % key.id, key.canonical().encode())
    store.put("wheels/%s/files/%s" % (key.id, name), b"x")

    with pytest.raises(ValueError):
        _cache.pull(store, key, str(tmp_path / "wheels" / "inner"))
    assert not (tmp_path / "escape.whl").exists()


def test_a_library_named_with_with_is_part_of_the_key():
    plain = _cache.compute_key("tilelang", gpus=["sm_89"])
    named = _cache.compute_key("tilelang", ["pyyaml"], gpus=["sm_89"])

    assert "lib:pyyaml" in named.fields
    assert plain.id != named.id


def test_the_command_line_pulls_what_it_pushed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_cache, "gpu_capabilities", lambda: ["sm_89"])
    first, second, store = (str(tmp_path / name) for name in ("first", "second", "store"))
    _fill(first, {"w.whl": b"wheel"})

    assert main(["cache", "push", "--name", "wheels", "--dir", first, "--store", store]) == 0
    assert main(["cache", "pull", "--name", "wheels", "--dir", second, "--store", store]) == 0

    assert "hit" in capsys.readouterr().out.splitlines()[-1]
    assert _read(second) == {"w.whl": b"wheel"}


def test_a_bucket_without_its_keys_is_refused_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("RAVEX_CACHE_ACCESS_KEY", raising=False)
    monkeypatch.delenv("RAVEX_CACHE_SECRET_KEY", raising=False)

    code = main(["cache", "pull", "--name", "wheels", "--dir", str(tmp_path), "--store", "s3://gpuzero-cache"])

    assert code == 1
    assert "RAVEX_CACHE_ACCESS_KEY" in capsys.readouterr().err


class SigningService:
    """A signing service and the bucket behind it, in one small HTTP server.

    ``POST /sign`` answers the protocol in :mod:`ravex._cache`; its URLs point
    back at ``/objects/<path>``, which keeps the bytes. Writes under ``ro/``
    are refused, as a service refuses a part that machines may only read.
    """

    def __init__(self, token="t0ken"):
        import threading
        import urllib.parse
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.token = token
        self.objects = {}
        self.object_auth = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body=b"", kind="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.headers.get("Authorization") != "Bearer " + service.token:
                    return self._send(401, b'{"detail": "no token"}')
                op, path = body["op"], body["path"]
                if op == "put" and path.startswith("ro/"):
                    return self._send(403, b'{"detail": "read-only"}')
                if op == "list":
                    paths = sorted(p for p in service.objects if p.startswith(path))
                    return self._send(200, json.dumps({"paths": paths}).encode())
                url = "%s/objects/%s" % (service.url, urllib.parse.quote(path))
                return self._send(200, json.dumps({"url": url}).encode())

            def _object(self):
                service.object_auth.append(self.headers.get("Authorization"))
                return urllib.parse.unquote(self.path[len("/objects/"):])

            def do_GET(self):
                path = self._object()
                if path not in service.objects:
                    return self._send(404)
                return self._send(200, service.objects[path], "application/octet-stream")

            def do_PUT(self):
                path = self._object()
                service.objects[path] = self.rfile.read(int(self.headers["Content-Length"]))
                return self._send(200)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def service():
    running = SigningService()
    yield running
    running.close()


def test_a_signed_store_pushes_and_pulls_without_a_bucket_key(tmp_path, monkeypatch, capsys, service):
    monkeypatch.setattr(_cache, "gpu_capabilities", lambda: ["sm_89"])
    monkeypatch.setenv("RAVEX_CACHE_TOKEN", service.token)
    for name in ("RAVEX_CACHE_ACCESS_KEY", "RAVEX_CACHE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    _fill(first, {"w.whl": b"wheel", "sub/k.cubin": b"kernel"})
    store = "sign+%s/sign" % service.url

    assert main(["cache", "push", "--name", "wheels", "--dir", first, "--store", store]) == 0
    assert main(["cache", "pull", "--name", "wheels", "--dir", second, "--store", store]) == 0

    assert "hit" in capsys.readouterr().out.splitlines()[-1]
    assert _read(second) == {"w.whl": b"wheel", "sub/k.cubin": b"kernel"}
    # The token goes to the service only: the bucket's URLs carry their own
    # signature, and a token sent there would be one more place it leaks.
    assert service.object_auth and not any(service.object_auth)


def test_a_signed_store_reports_a_missing_object_as_none(service):
    store = _cache.SignedStore(service.url + "/sign", service.token)
    assert store.get("nothing/here") is None


def test_a_refused_write_fails_the_push_and_says_why(tmp_path, monkeypatch, capsys, service):
    monkeypatch.setattr(_cache, "gpu_capabilities", lambda: ["sm_89"])
    monkeypatch.setenv("RAVEX_CACHE_TOKEN", service.token)
    directory = str(tmp_path / "d")
    _fill(directory, {"w.whl": b"wheel"})

    code = main(["cache", "push", "--name", "wheels", "--dir", directory, "--store", "sign+%s/sign" % service.url])
    assert code == 0  # wheels/ is writable here

    store = _cache.SignedStore(service.url + "/sign", service.token)
    with pytest.raises(OSError, match="403"):
        store.put("ro/wheels/x", b"x")


def test_a_signing_service_without_a_token_is_refused_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("RAVEX_CACHE_TOKEN", raising=False)

    code = main(["cache", "pull", "--name", "wheels", "--dir", str(tmp_path), "--store", "sign+https://example.invalid/sign"])

    assert code == 1
    assert "RAVEX_CACHE_TOKEN" in capsys.readouterr().err
