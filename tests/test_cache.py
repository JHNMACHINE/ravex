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


def test_a_bucket_is_refused_with_a_reason_until_it_is_supported(tmp_path, capsys):
    code = main(["cache", "pull", "--name", "wheels", "--dir", str(tmp_path), "--store", "s3://gpuzero-cache"])

    assert code == 1
    assert "bucket" in capsys.readouterr().err
