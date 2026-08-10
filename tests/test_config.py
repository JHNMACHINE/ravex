import textwrap

from ravex._config import RavexConfig, find_config_file


def test_defaults_are_sane():
    config = RavexConfig()
    assert config.enabled
    assert config.backend == "moonclip"
    assert config.checkpoint_every == 500
    assert config.storage.type == "local"


def test_yaml_is_loaded(tmp_path, monkeypatch):
    path = tmp_path / "ravex.yaml"
    path.write_text(
        textwrap.dedent(
            """
            checkpoint_every: 42
            backend: torch_save
            keep_last: 2
            storage:
              type: local
              path: ./ckpt
            frameworks:
              auto_detect: false
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAVEX_CONFIG", str(path))

    config = RavexConfig.load()
    assert config.checkpoint_every == 42
    assert config.backend == "torch_save"
    assert config.keep_last == 2
    assert config.storage.path == "./ckpt"
    assert config.framework_auto_detect is False
    assert config.source == str(path)


def test_env_beats_yaml(tmp_path, monkeypatch):
    path = tmp_path / "ravex.yaml"
    path.write_text("checkpoint_every: 42\n", encoding="utf-8")
    monkeypatch.setenv("RAVEX_CONFIG", str(path))
    monkeypatch.setenv("RAVEX_CHECKPOINT_EVERY", "7")

    assert RavexConfig.load().checkpoint_every == 7


def test_config_is_found_from_a_subdirectory(tmp_path, monkeypatch):
    (tmp_path / "ravex.yaml").write_text("enabled: true\n", encoding="utf-8")
    nested = tmp_path / "src" / "training"
    nested.mkdir(parents=True)

    monkeypatch.delenv("RAVEX_CONFIG", raising=False)
    monkeypatch.chdir(nested)

    assert find_config_file() == tmp_path / "ravex.yaml"


def test_remote_storage_without_a_bucket_degrades_to_local(monkeypatch):
    monkeypatch.setenv("RAVEX_STORAGE_TYPE", "s3")
    config = RavexConfig.load()
    # Misconfiguration must not take the run down: fall back to local disk.
    assert config.storage.type == "local"


def test_credentials_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("RAVEX_STORAGE_TYPE", "r2")
    monkeypatch.setenv("RAVEX_STORAGE_BUCKET", "runs")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sk")

    config = RavexConfig.load()
    assert config.storage.is_remote
    assert (config.storage.access_key, config.storage.secret_key) == ("ak", "sk")


def test_run_id_becomes_the_storage_prefix(monkeypatch):
    monkeypatch.setenv("RAVEX_RUN_ID", "run-123")
    assert RavexConfig.load().storage.prefix == "run-123"
