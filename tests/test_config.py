import textwrap

import pytest

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


def test_a_truncated_config_does_not_stop_ravex(tmp_path, monkeypatch):
    """One interrupted write must not silently disable checkpointing.

    `checkpoint_every:` with nothing after it parses as None, which used to
    raise while normalising. The autoloader swallows that by design, so the run
    trained on with no checkpoints, no log file - logging is configured after
    the config loads - and no way to tell.
    """
    path = tmp_path / "ravex.yaml"
    path.write_text("checkpoint_every:", encoding="utf-8")  # truncated mid-write
    monkeypatch.setenv("RAVEX_CONFIG", str(path))

    config = RavexConfig.load()

    assert config.checkpoint_every == 500, "should fall back to the default"
    assert config.problems, "the substitution must be reported, not hidden"
    assert "checkpoint_every" in config.problems[0]


@pytest.mark.parametrize(
    "yaml_text, field, expected",
    [
        ("keep_last: many\n", "keep_last", 5),
        ("checkpoint_every: []\n", "checkpoint_every", 500),
        ("enabled: maybe\n", "enabled", True),
        ("backend: 7\n", "backend", "moonclip"),
    ],
)
def test_nonsense_values_fall_back_to_defaults(
    tmp_path, monkeypatch, yaml_text, field, expected
):
    path = tmp_path / "ravex.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("RAVEX_CONFIG", str(path))

    config = RavexConfig.load()

    assert getattr(config, field) == expected
    assert any(field in problem for problem in config.problems)


def test_run_id_becomes_the_storage_prefix(monkeypatch):
    monkeypatch.setenv("RAVEX_RUN_ID", "run-123")
    assert RavexConfig.load().storage.prefix == "run-123"


def test_sharded_checkpoints_defaults_to_gathering():
    """Per-rank checkpoints give up resuming at a different number of GPUs.
    That is not something to acquire by upgrading."""
    assert RavexConfig.load().sharded_checkpoints == "gather"


def test_sharded_checkpoints_reads_yaml_and_env(tmp_path, monkeypatch):
    path = tmp_path / "ravex.yaml"
    path.write_text("sharded_checkpoints: per_rank\n", encoding="utf-8")
    monkeypatch.setenv("RAVEX_CONFIG", str(path))
    assert RavexConfig.load().sharded_checkpoints == "per_rank"

    monkeypatch.setenv("RAVEX_SHARDED_CHECKPOINTS", "PER_RANK")
    assert RavexConfig.load().sharded_checkpoints == "per_rank"


def test_an_unknown_sharded_mode_falls_back_and_says_so(tmp_path, monkeypatch):
    path = tmp_path / "ravex.yaml"
    path.write_text("sharded_checkpoints: per-rank\n", encoding="utf-8")
    monkeypatch.setenv("RAVEX_CONFIG", str(path))

    config = RavexConfig.load()
    assert config.sharded_checkpoints == "gather", "a typo must not silently shard"
    assert any("sharded_checkpoints" in problem for problem in config.problems)


def test_keep_base_in_memory_reads_yaml_and_env(tmp_path, monkeypatch):
    assert RavexConfig().keep_base_in_memory, "must default to Moonclip's default"

    path = tmp_path / "ravex.yaml"
    path.write_text("keep_base_in_memory: false", encoding="utf-8")
    monkeypatch.setenv("RAVEX_CONFIG", str(path))
    assert RavexConfig.load().keep_base_in_memory is False

    # The env var is what an A/B run on a rented box actually uses: two
    # otherwise identical invocations, one variable between them.
    monkeypatch.setenv("RAVEX_KEEP_BASE_IN_MEMORY", "true")
    assert RavexConfig.load().keep_base_in_memory is True


def test_a_junk_keep_base_value_falls_back_and_says_so(tmp_path, monkeypatch):
    path = tmp_path / "ravex.yaml"
    path.write_text("keep_base_in_memory: maybe", encoding="utf-8")
    monkeypatch.setenv("RAVEX_CONFIG", str(path))

    config = RavexConfig.load()
    assert config.keep_base_in_memory is True, "a typo must not drop the base"
    assert any("keep_base_in_memory" in problem for problem in config.problems)


# ─── save_dtype ─────────────────────────────────────────────────────


def _normalized(value):
    config = RavexConfig()
    config.save_dtype = value
    config._normalize()
    return config


def test_save_dtype_is_off_by_default():
    """It changes the numbers a resumed run gets back, so it is opt-in and
    stays opt-in across an upgrade."""
    assert RavexConfig().save_dtype is None
    assert RavexConfig().resolve_save_dtype() is None


def test_a_bare_dtype_applies_to_everything():
    config = _normalized("bf16")
    assert config.resolve_save_dtype() == "bf16"
    assert not config.problems


def test_none_is_spelled_out_as_nothing_configured():
    """`save_dtype: none` and no `save_dtype:` at all say the same thing, and
    should reach Moonclip the same way — as an argument left out."""
    assert _normalized("none").resolve_save_dtype() is None


def test_a_component_expands_to_every_place_it_is_written():
    """The reason components exist rather than raw patterns only.

    An optimizer lives under `ravex/optimizers/` when it is not sharded and
    under `ravex/sharded/<key>/optimizer/` when it is. One name, two places.
    """
    assert _normalized({"optimizer": "bf16"}).resolve_save_dtype() == {
        "ravex/optimizers/*": "bf16",
        "ravex/sharded/*/optimizer/*": "bf16",
    }
    assert _normalized({"model": "fp16"}).resolve_save_dtype() == {
        "ravex/models/*": "fp16",
        "ravex/sharded/*/model/*": "fp16",
    }


def test_expansion_keeps_the_order_the_rules_were_written_in():
    """Moonclip applies the first rule that matches, so the exception has to
    stay ahead of the catch-all after expanding — otherwise
    `{model: none, "*": bf16}` quietly becomes "cast everything"."""
    resolved = _normalized({"model": "none", "*": "bf16"}).resolve_save_dtype()
    assert list(resolved) == [
        "ravex/models/*",
        "ravex/sharded/*/model/*",
        "*",
    ]
    assert resolved["ravex/models/*"] == "none"
    assert resolved["*"] == "bf16"


def test_a_raw_pattern_passes_straight_through():
    """The escape hatch for anything the component names do not cover."""
    config = _normalized({"ravex/sharded/*/optimizer/*": "fp8"})
    assert config.resolve_save_dtype() == {"ravex/sharded/*/optimizer/*": "fp8"}
    assert not config.problems


@pytest.mark.parametrize(
    "value,complaint",
    [
        ("bfloat", "not a dtype"),
        ({"optimizer": "bfloat"}, "not a dtype"),
        ({"optimiser": "bf16"}, "neither a component"),
        ({"weights": "bf16"}, "neither a component"),
        (17, "neither a dtype nor a mapping"),
    ],
)
def test_an_unusable_save_dtype_is_dropped_and_recorded(value, complaint):
    """Moonclip refuses a bad target too, but at construction — which under
    Ravex is mid-run, inside the `except` that falls back to `torch.save`. A
    typo would cost the run its Moonclip checkpointing. Here it lands in
    `problems` before anything starts."""
    config = _normalized(value)
    assert config.resolve_save_dtype() is None
    assert any(complaint in p for p in config.problems), config.problems


def test_one_bad_rule_does_not_take_the_good_ones_with_it():
    config = _normalized({"optimizer": "bf16", "weights": "none"})
    assert config.resolve_save_dtype() == {
        "ravex/optimizers/*": "bf16",
        "ravex/sharded/*/optimizer/*": "bf16",
    }
    assert len(config.problems) == 1


def test_save_dtype_reads_yaml_and_env(tmp_path, monkeypatch):
    path = tmp_path / "ravex.yaml"
    path.write_text(
        textwrap.dedent(
            """
            save_dtype:
              optimizer: bf16
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAVEX_CONFIG", str(path))
    monkeypatch.delenv("RAVEX_SAVE_DTYPE", raising=False)
    assert RavexConfig.load().save_dtype == {"optimizer": "bf16"}

    # A bare dtype in the environment.
    monkeypatch.setenv("RAVEX_SAVE_DTYPE", "fp16")
    assert RavexConfig.load().save_dtype == "fp16"

    # Rules in the environment, ordered by where they sit in the string.
    monkeypatch.setenv("RAVEX_SAVE_DTYPE", "model:none,optimizer:bf16")
    loaded = RavexConfig.load()
    assert list(loaded.save_dtype) == ["model", "optimizer"]
    assert loaded.save_dtype == {"model": "none", "optimizer": "bf16"}


def test_a_junk_save_dtype_in_the_environment_falls_back_and_says_so(monkeypatch):
    monkeypatch.setenv("RAVEX_SAVE_DTYPE", "optimizer:bfloat")
    config = RavexConfig.load()
    assert config.resolve_save_dtype() is None
    assert any("not a dtype" in p for p in config.problems), config.problems
