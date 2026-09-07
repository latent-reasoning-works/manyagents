"""Execute harmless scripts to verify CellForge's installation binding."""

from pathlib import Path

import pytest

from manyagents.adapters.cellforge_adapter import CellForgeAdapter


@pytest.mark.parametrize("source", ["argument", "environment"])
async def test_task_working_dir_cannot_redirect_main(source, monkeypatch, tmp_path):
    install = tmp_path / "installed"
    task_dir = tmp_path / "task"
    install.mkdir()
    task_dir.mkdir()
    (install / "main.py").write_text("print('validated installation')\n")
    (task_dir / "main.py").write_text("print('substituted executable')\n")
    data = tmp_path / "data.h5ad"
    data.touch()
    monkeypatch.chdir(install)
    if source == "environment":
        monkeypatch.setenv("CELLFORGE_PATH", ".")
        adapter = CellForgeAdapter()
    else:
        adapter = CellForgeAdapter(cellforge_path=".")
    result = await adapter.run({"working_dir": str(task_dir), "data_file": str(data)}, {})
    assert result["success"], result
    assert result["output_files"]["stdout"].read_text().strip() == "validated installation"
    assert adapter.main_script == (install / "main.py").resolve()
    # Version probing must keep the same binding even if the parent later changes cwd.
    monkeypatch.chdir(task_dir)
    assert await adapter._get_cellforge_version() == "validated installation"


@pytest.mark.parametrize("env_value", [None, ""])
def test_cellforge_requires_explicit_installation(env_value, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "main.py").write_text("print('unconfigured')\n")
    monkeypatch.delenv("CELLFORGE_PATH", raising=False)
    if env_value is not None:
        monkeypatch.setenv("CELLFORGE_PATH", env_value)
    with pytest.raises(ValueError, match="cellforge_path.*CELLFORGE_PATH"):
        CellForgeAdapter()


async def test_cellforge_rejects_main_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "main.py").mkdir()
    data = tmp_path / "data.h5ad"
    data.touch()
    adapter = CellForgeAdapter(cellforge_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="CellForge main.py"):
        await adapter.run({"data_file": str(data)}, {})


def test_cellforge_binds_symlink_target(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "entry.py"
    script.touch()
    (tmp_path / "main.py").symlink_to(script)
    adapter = CellForgeAdapter(cellforge_path=".")
    assert adapter.main_script == script.resolve()
    assert Path(adapter._build_command("all", tmp_path / "data.h5ad")[0]).is_absolute()
