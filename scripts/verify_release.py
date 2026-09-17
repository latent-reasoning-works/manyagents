"""Verify both release install paths outside the checkout (requires network and uv)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import zipfile


INSTALLED_PROBE = r'''
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

import manyagents
from hydra import compose, initialize_config_module

expected = json.loads(sys.argv[1])
assert Path(manyagents.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert manyagents.__version__ == metadata.version("manyagents") == expected["version"]
package = resources.files("manyagents")
for name, digest in expected["configs"].items():
    assert hashlib.sha256(package.joinpath(name).read_bytes()).hexdigest() == digest, name
with initialize_config_module(version_base=None, config_module="manyagents.configs"):
    for name in expected["experiments"]:
        compose(config_name="main", overrides=[f"experiment={name}"])
entries = [entry for entry in metadata.distribution("manyagents").entry_points
           if entry.group == "console_scripts"]
assert {entry.name for entry in entries} == set(expected["scripts"])
for entry in entries:
    executable = Path(sysconfig.get_path("scripts")) / entry.name
    if sys.platform == "win32":
        executable = executable.with_suffix(".exe")
    subprocess.run([str(executable), "--help"], check=True)
print(f"Verified installed manyagents {manyagents.__version__}: "
      f"{len(expected['configs'])} Hydra configs and {len(entries)} console scripts")
'''


def verify_install(wheel, directory, expected, env):
    """Install a wheel with dependencies into a new venv, then probe its contents."""
    directory.mkdir()
    venv = directory / 'venv'
    subprocess.run(['uv', 'venv', '--python', sys.executable, str(venv)],
                   check=True, cwd=directory, env=env)
    python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    subprocess.run(['uv', 'pip', 'install', '--python', str(python), str(wheel)],
                   check=True, cwd=directory, env=env)
    subprocess.run(['uv', 'pip', 'check', '--python', str(python)],
                   check=True, cwd=directory, env=env)
    subprocess.run([str(python), '-I', '-c', INSTALLED_PROBE, json.dumps(expected)],
                   check=True, cwd=directory, env=env)


def main():
    """Verify the wheel, rebuild the sdist, and verify the rebuilt wheel separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dist', type=Path, help='Directory containing one wheel and one sdist')
    args = parser.parse_args()
    wheel, = args.dist.resolve().glob('*.whl')
    sdist, = args.dist.resolve().glob('*.tar.gz')
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    package = root / 'manyagents'
    expected = {
        'version': project['version'],
        'scripts': list(project['scripts']),
        'configs': {
            path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((package / 'configs').rglob('*.yaml'))
        },
        'experiments': sorted(path.stem for path in (package / 'configs/experiment').glob('*.yaml')),
    }
    assert expected['configs'] and expected['experiments']
    # Do not inherit an editable source path or a developer's active environment.
    env = {key: value for key, value in os.environ.items()
           if key not in {'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'}}
    with tempfile.TemporaryDirectory(prefix='manyagents-release-') as scratch:
        directory = Path(scratch)
        verify_install(wheel, directory / 'wheel', expected, env)
        rebuilt = directory / 'rebuilt'
        subprocess.run(['uv', 'build', str(sdist), '--wheel', '--python', sys.executable,
                        '--out-dir', str(rebuilt)], check=True, cwd=directory, env=env)
        rebuilt_wheel, = rebuilt.glob('*.whl')
        # The two paths must deliver the same package and runtime data.
        with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(rebuilt_wheel) as other:
            original_files = {name: original.read(name) for name in original.namelist()
                              if name.startswith('manyagents/')}
            rebuilt_files = {name: other.read(name) for name in other.namelist()
                             if name.startswith('manyagents/')}
            assert original_files == rebuilt_files
        verify_install(rebuilt_wheel, directory / 'sdist', expected, env)


if __name__ == '__main__':
    main()
