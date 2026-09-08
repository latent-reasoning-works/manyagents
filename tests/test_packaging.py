"""Phase 2 packaging contracts; real install probes run from git archives."""

import builtins
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from types import ModuleType
from unittest.mock import MagicMock
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_lock_uses_registry_source():
    lock = tomllib.loads((ROOT / 'uv.lock').read_text())
    package, = [p for p in lock['package'] if p['name'] == 'manylatents']
    assert package['source'] == {'registry': 'https://pypi.org/simple'}
    assert package['version'].startswith('0.1.')


def test_wheel_builds_and_has_no_hook(tmp_path):
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert 'custom' not in project['tool']['hatch']['build'].get('hooks', {})
    assert not (ROOT / 'hatch_build.py').exists()
    subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(tmp_path)], cwd=ROOT, check=True)
    wheel, = tmp_path.glob('*.whl')
    with zipfile.ZipFile(wheel) as archive:
        assert 'manyagents/inference.py' in archive.namelist()
        assert not any('hatch_build' in name for name in archive.namelist())
        version = project['project']['version']
        metadata = archive.read(f'manyagents-{version}.dist-info/METADATA').decode()
        assert 'License-Expression: MIT' in metadata
        assert archive.read(f'manyagents-{version}.dist-info/licenses/LICENSE').startswith(b'MIT License')
        assert 'Provides-Extra: traces' in metadata
        assert 'Provides-Extra: wandb' in metadata
        assert 'Provides-Extra: docs' not in metadata


@pytest.fixture
def registry_modules(tmp_path, monkeypatch):
    from manyagents.adapters import metric_registry as registry
    from manyagents.adapters import _generate_metric_registry as generator

    package = tmp_path / 'manylatents'
    configs = package / 'configs' / 'metrics'
    configs.mkdir(parents=True)
    (configs / 'example.yaml').write_text('example:\n  at: embedding\n  _target_: manylatents.metrics.Example\n')
    stub = ModuleType('manylatents')
    stub.__file__ = str(package / '__init__.py')
    monkeypatch.setitem(sys.modules, 'manylatents', stub)
    monkeypatch.setattr(generator, 'discover_manylatents_extensions', lambda: [])
    return registry, generator


def test_registry_init_with_readonly_package_dir(tmp_path, monkeypatch, registry_modules):
    registry, generator = registry_modules
    package = tmp_path / 'manyagents' / 'adapters'
    data = package / 'data'
    data.mkdir(parents=True)
    monkeypatch.setattr(registry, '__file__', str(package / 'metric_registry.py'))
    data.chmod(0o555)
    try:
        instance = registry.MetricRegistry()
        assert 'example' in instance
        assert instance.registry_path is None
        assert list(data.iterdir()) == []
    finally:
        data.chmod(0o755)


def test_generator_defaults_to_memory(registry_modules):
    _, generator = registry_modules
    result = generator.generate_metric_registry()
    assert 'example' in result['metrics']


def test_registry_explicit_path_persists(tmp_path, registry_modules):
    registry, _ = registry_modules
    path = tmp_path / 'configured' / 'registry.json'
    instance = registry.MetricRegistry(path)
    assert json.loads(path.read_text())['metrics'] == instance.metrics
    assert registry.MetricRegistry(path, auto_regenerate=False).metrics == instance.metrics


def test_registry_explicit_write_failure_keeps_generated_data(tmp_path, registry_modules):
    registry, _ = registry_modules
    directory = tmp_path / 'readonly'
    directory.mkdir()
    directory.chmod(0o555)
    try:
        instance = registry.MetricRegistry(directory / 'registry.json')
        assert 'example' in instance
    finally:
        directory.chmod(0o755)


def block_import(monkeypatch, prefix):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == prefix or name.startswith(prefix + '.'):
            raise ModuleNotFoundError(f'No module named {name!r}', name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)


@pytest.mark.parametrize('dtype', [None, 'float32'])
def test_load_model_without_manylatents(monkeypatch, dtype):
    import torch
    from manyagents import inference

    transformers = ModuleType('transformers')
    transformers.AutoTokenizer = MagicMock()
    transformers.AutoModelForCausalLM = MagicMock()
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    block_import(monkeypatch, 'manylatents')
    model, tok, trainer = inference.load_model(
        'fixture', device_map='cpu', dtype=dtype,
        trust_remote_code=False, attn_implementation='eager',
    )
    assert trainer is None
    assert model is transformers.AutoModelForCausalLM.from_pretrained.return_value.eval.return_value
    assert tok is transformers.AutoTokenizer.from_pretrained.return_value
    transformers.AutoTokenizer.from_pretrained.assert_called_once_with('fixture', trust_remote_code=False)
    transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
        'fixture', torch_dtype=dtype or torch.bfloat16, device_map='cpu',
        trust_remote_code=False, attn_implementation='eager',
    )


def test_hooks_error_names_traces_extra(monkeypatch):
    from manyagents.inference import resolve_layer_specs
    block_import(monkeypatch, 'manylatents')
    with pytest.raises(ImportError, match='--extra traces'):
        resolve_layer_specs('fixture')


def test_vllm_error_names_vllm_extra(monkeypatch):
    from manyagents.inference import get_vllm_engine
    block_import(monkeypatch, 'vllm')
    with pytest.raises(ImportError, match='--extra vllm'):
        get_vllm_engine('fixture/packaging')


@pytest.mark.parametrize('dependency', ['manylatents', 'scipy'])
@pytest.mark.parametrize('segmenter', ['segment_by_velocity', 'segment_hybrid'])
def test_geometry_error_names_traces_extra(monkeypatch, dependency, segmenter):
    import numpy as np
    from manyagents import inference

    # A character tokenizer keeps the hybrid thinking range nonempty.
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, **kwargs: list(range(len(text)))
    signal = ModuleType('scipy.signal')
    signal.find_peaks = MagicMock()
    monkeypatch.setitem(sys.modules, 'scipy.signal', signal)
    block_import(monkeypatch, dependency)
    text = '<think>abcdefghijk</think>answer'
    with pytest.raises(ImportError, match='--extra traces'):
        getattr(inference, segmenter)(text, tokenizer, np.ones((len(text), 1, 3)))


def test_registry_missing_dependency_names_traces_extra(monkeypatch):
    from manyagents.adapters.metric_registry import MetricRegistry
    block_import(monkeypatch, 'manylatents')
    with pytest.raises(ImportError, match='--extra traces'):
        MetricRegistry()
