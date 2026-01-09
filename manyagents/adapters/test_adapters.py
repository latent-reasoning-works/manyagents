"""Tests for adapter implementations.

Uses pytest parameterization to test common functionality across all adapters.
"""

import asyncio
import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from .base import AgentAdapter, AdapterResult, AdapterConfig
from manyagents.utils.helpers import (
    run_subprocess, SubprocessResult, get_python_executable,
    truncate_string, parse_json_safe, find_executable
)
from .openai_adapter import OpenAIAdapter
from .cellforge_adapter import CellForgeAdapter
from .biomni_adapter import BiomniAdapter
from .kosmos_adapter import KosmosAdapter


# =============================================================================
# Fixtures
# =============================================================================

class ConcreteAdapter(AgentAdapter):
    """Minimal concrete adapter for base class testing."""
    async def run(self, task_config, input_files):
        return self.success_response("Done", {}, {})


@pytest.fixture
def base_adapter(tmp_path):
    """Create base adapter with temp output dir."""
    config = AdapterConfig(output_base_dir=tmp_path)
    return ConcreteAdapter("test", config)


@pytest.fixture
def openai_adapter():
    """Create OpenAI adapter with mock API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        return OpenAIAdapter()


@pytest.fixture
def cellforge_adapter(tmp_path):
    """Create CellForge adapter with mock installation."""
    (tmp_path / "main.py").write_text("# mock")
    (tmp_path / "data.h5ad").write_text("mock")
    return CellForgeAdapter(cellforge_path=str(tmp_path))


@pytest.fixture
def biomni_adapter(tmp_path, monkeypatch):
    """Create Biomni adapter with mock data path."""
    data_path = tmp_path / "biomni_data"
    data_path.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return BiomniAdapter(data_path=str(data_path))


@pytest.fixture
def kosmos_adapter(tmp_path):
    """Create Kosmos adapter with mock Python path."""
    mock_python = tmp_path / "python"
    mock_python.write_text("#!/bin/bash\necho mock")
    mock_python.chmod(0o755)
    return KosmosAdapter(
        kosmos_python=str(mock_python),
        kosmos_path=str(tmp_path)
    )


@pytest.fixture(params=["openai", "cellforge", "biomni", "kosmos"])
def any_adapter(request, tmp_path, monkeypatch):
    """Parameterized fixture returning any adapter type."""
    if request.param == "openai":
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        return OpenAIAdapter()
    elif request.param == "cellforge":
        (tmp_path / "main.py").write_text("# mock")
        return CellForgeAdapter(cellforge_path=str(tmp_path))
    elif request.param == "biomni":
        data_path = tmp_path / "biomni_data"
        data_path.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        return BiomniAdapter(data_path=str(data_path))
    else:  # kosmos
        mock_python = tmp_path / "python"
        mock_python.write_text("#!/bin/bash\necho mock")
        mock_python.chmod(0o755)
        return KosmosAdapter(kosmos_python=str(mock_python), kosmos_path=str(tmp_path))


# =============================================================================
# Base Adapter Tests
# =============================================================================

class TestAdapterBase:
    """Test base adapter functionality."""

    def test_success_response_structure(self, base_adapter):
        """Test success_response returns correct structure."""
        result = base_adapter.success_response("OK", {"file": Path("x")}, {"key": "val"})
        assert result["success"] is True
        assert result["summary"] == "OK"
        assert "file" in result["output_files"]
        assert result["metadata"]["key"] == "val"

    def test_error_response_structure(self, base_adapter):
        """Test error_response returns correct structure."""
        result = base_adapter.error_response("Failed", "timeout", "details")
        assert result["success"] is False
        assert result["metadata"]["error"] == "timeout"
        assert result["metadata"]["details"] == "details"

    def test_error_response_merges_metadata(self, base_adapter):
        """Test error_response merges additional metadata."""
        result = base_adapter.error_response("Failed", "err", metadata={"extra": 123})
        assert result["metadata"]["error"] == "err"
        assert result["metadata"]["extra"] == 123

    def test_output_dir_created_on_access(self, base_adapter, tmp_path):
        """Test output_dir property creates directory."""
        output = base_adapter.output_dir
        assert output.exists()
        assert output == tmp_path / "test"

    def test_save_text_output(self, base_adapter):
        """Test saving text to file."""
        path = base_adapter.save_text_output("content", "out.txt")
        assert path.read_text() == "content"

    def test_save_json_output(self, base_adapter):
        """Test saving JSON to file."""
        path = base_adapter.save_json_output({"a": 1}, "out.json")
        assert json.loads(path.read_text()) == {"a": 1}


# =============================================================================
# Utilities Tests
# =============================================================================

class TestUtils:
    """Test shared utility functions."""

    @pytest.mark.asyncio
    async def test_subprocess_success(self):
        result = await run_subprocess(["echo", "hi"])
        assert result.exit_code == 0
        assert "hi" in result.stdout

    @pytest.mark.asyncio
    async def test_subprocess_timeout(self):
        result = await run_subprocess(["sleep", "10"], timeout=0.1)
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_subprocess_not_found(self):
        result = await run_subprocess(["nonexistent_cmd_xyz"])
        assert result.exit_code == -1

    def test_truncate_short(self):
        assert truncate_string("hi", 10) == "hi"

    def test_truncate_long(self):
        assert truncate_string("hello world", 5) == "hello..."

    def test_parse_json_valid(self):
        data, err = parse_json_safe('{"x": 1}')
        assert data == {"x": 1}
        assert err is None

    def test_parse_json_invalid(self):
        data, err = parse_json_safe('bad')
        assert data is None
        assert err is not None

    def test_get_python_executable(self):
        assert "python" in get_python_executable().lower()


# =============================================================================
# Common Adapter Behavior (Parameterized)
# =============================================================================

class TestAdapterCommon:
    """Test behavior common to all adapters."""

    def test_has_name(self, any_adapter):
        """All adapters have a name."""
        assert isinstance(any_adapter.name, str)
        assert len(any_adapter.name) > 0

    def test_inherits_base(self, any_adapter):
        """All adapters inherit from AgentAdapter."""
        assert isinstance(any_adapter, AgentAdapter)

    def test_has_run_method(self, any_adapter):
        """All adapters have async run method."""
        assert asyncio.iscoroutinefunction(any_adapter.run)

    def test_has_error_response(self, any_adapter):
        """All adapters have error_response from base."""
        result = any_adapter.error_response("test error", "test_type")
        assert result["success"] is False
        assert result["metadata"]["error"] == "test_type"

    def test_has_success_response(self, any_adapter):
        """All adapters have success_response from base."""
        result = any_adapter.success_response("test ok")
        assert result["success"] is True


# =============================================================================
# OpenAI-Specific Tests
# =============================================================================

class TestOpenAI:
    """OpenAI adapter specific tests."""

    def test_init_with_key(self, openai_adapter):
        assert openai_adapter.client is not None

    def test_init_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            adapter = OpenAIAdapter()
            assert adapter.client is None

    @pytest.mark.asyncio
    async def test_missing_prompt(self, openai_adapter):
        result = await openai_adapter.run(task_config={}, input_files={})
        assert result["success"] is False
        assert "prompt" in result["summary"].lower()

    @pytest.mark.asyncio
    async def test_missing_key_at_runtime(self):
        with patch.dict(os.environ, {}, clear=True):
            adapter = OpenAIAdapter()
            result = await adapter.run(task_config={"prompt": "hi"}, input_files={})
            assert result["metadata"]["error"] == "missing_api_key"

    @pytest.mark.asyncio
    async def test_successful_call(self, openai_adapter, tmp_path):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "response"
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)

        with patch.object(openai_adapter.client.chat.completions, 'create',
                          new_callable=AsyncMock, return_value=mock_resp):
            os.chdir(tmp_path)
            result = await openai_adapter.run(task_config={"prompt": "test"}, input_files={})
            assert result["success"] is True
            assert result["metadata"]["tokens_used"]["total_tokens"] == 15


# =============================================================================
# CellForge-Specific Tests
# =============================================================================

class TestCellForge:
    """CellForge adapter specific tests."""

    def test_valid_phases(self):
        assert "task_analysis" in CellForgeAdapter.VALID_PHASES
        assert "all" in CellForgeAdapter.VALID_PHASES

    def test_init_env_path(self, monkeypatch):
        monkeypatch.setenv("CELLFORGE_PATH", "/env")
        adapter = CellForgeAdapter()
        assert adapter.cellforge_path == Path("/env")

    @pytest.mark.asyncio
    async def test_invalid_phase(self, cellforge_adapter, tmp_path):
        with pytest.raises(ValueError, match="Invalid phase"):
            await cellforge_adapter.run(
                task_config={"phase": "bad", "data_file": str(tmp_path / "data.h5ad")},
                input_files={}
            )

    @pytest.mark.asyncio
    async def test_missing_data(self, cellforge_adapter):
        with pytest.raises(ValueError, match="data_file"):
            await cellforge_adapter.run(task_config={}, input_files={})

    @pytest.mark.asyncio
    async def test_successful_run(self, cellforge_adapter, tmp_path):
        with patch('asyncio.create_subprocess_exec') as mock:
            proc = AsyncMock()
            proc.communicate.return_value = (b"ok", b"")
            proc.returncode = 0
            mock.return_value = proc

            result = await cellforge_adapter.run(
                task_config={"data_file": str(tmp_path / "data.h5ad"), "working_dir": str(tmp_path)},
                input_files={}
            )
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_timeout(self, cellforge_adapter, tmp_path):
        with patch('asyncio.create_subprocess_exec') as mock:
            proc = AsyncMock()
            proc.communicate.side_effect = asyncio.TimeoutError()
            proc.kill = MagicMock()
            proc.wait = AsyncMock()
            mock.return_value = proc

            result = await cellforge_adapter.run(
                task_config={"data_file": str(tmp_path / "data.h5ad"), "working_dir": str(tmp_path), "timeout": 1},
                input_files={}
            )
            assert result["success"] is False
            assert "timed out" in result["summary"].lower()

    def test_build_command_phase(self, cellforge_adapter):
        cmd = cellforge_adapter._build_command("task_analysis", Path("/data.h5ad"))
        assert "--phase" in cmd
        assert "task_analysis" in cmd

    def test_build_command_all(self, cellforge_adapter):
        cmd = cellforge_adapter._build_command("all", Path("/data.h5ad"))
        assert "--phase" not in cmd


# =============================================================================
# Biomni-Specific Tests
# =============================================================================

class TestBiomni:
    """Biomni adapter specific tests."""

    def test_init_with_path(self, biomni_adapter):
        assert biomni_adapter.data_path.exists()

    def test_init_env_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BIOMNI_DATA_PATH", str(tmp_path))
        adapter = BiomniAdapter()
        assert adapter.data_path == tmp_path

    @pytest.mark.asyncio
    async def test_missing_task(self, biomni_adapter):
        result = await biomni_adapter.run(task_config={}, input_files={})
        assert result["success"] is False
        assert result["metadata"]["error"] == "missing_parameter"

    @pytest.mark.asyncio
    async def test_missing_api_key(self, tmp_path):
        with patch.dict(os.environ, {}, clear=True):
            adapter = BiomniAdapter(data_path=str(tmp_path))
            result = await adapter.run(
                task_config={"task": "test task"},
                input_files={}
            )
            assert result["metadata"]["error"] == "missing_api_key"

    @pytest.mark.asyncio
    async def test_successful_run(self, biomni_adapter, tmp_path):
        import sys

        # Create mock module structure
        mock_a1_class = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_instance.go.return_value = "Analysis complete"
        mock_a1_class.return_value = mock_agent_instance

        mock_agent_module = MagicMock()
        mock_agent_module.A1 = mock_a1_class

        mock_biomni = MagicMock()
        mock_biomni.agent = mock_agent_module

        with patch.dict(sys.modules, {
            "biomni": mock_biomni,
            "biomni.agent": mock_agent_module
        }):
            os.chdir(tmp_path)
            result = await biomni_adapter.run(
                task_config={"task": "Analyze dataset"},
                input_files={}
            )
            assert result["success"] is True
            assert "execution_time" in result["metadata"]

    @pytest.mark.asyncio
    async def test_import_error(self, tmp_path, monkeypatch):
        import sys

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        adapter = BiomniAdapter(data_path=str(tmp_path))

        # Clear any cached biomni import and force ImportError
        modules_to_remove = [k for k in sys.modules if k.startswith("biomni")]
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)

        with patch.dict(sys.modules, {"biomni": None}):
            result = await adapter.run(
                task_config={"task": "test"},
                input_files={}
            )
            assert result["success"] is False
            assert result["metadata"]["error"] == "import_error"


# =============================================================================
# Kosmos-Specific Tests
# =============================================================================

class TestKosmos:
    """Kosmos adapter specific tests."""

    def test_init_with_paths(self, kosmos_adapter):
        assert kosmos_adapter.kosmos_python is not None

    def test_init_env_paths(self, monkeypatch, tmp_path):
        mock_python = tmp_path / "python3"
        mock_python.write_text("#!/bin/bash\necho mock")
        mock_python.chmod(0o755)

        monkeypatch.setenv("KOSMOS_PYTHON", str(mock_python))
        monkeypatch.setenv("KOSMOS_PATH", str(tmp_path))
        adapter = KosmosAdapter()
        assert adapter.kosmos_python == str(mock_python)
        assert adapter.kosmos_path == tmp_path

    @pytest.mark.asyncio
    async def test_missing_python_path(self, tmp_path):
        adapter = KosmosAdapter(kosmos_path=str(tmp_path))
        result = await adapter.run(
            task_config={"research_question": "test"},
            input_files={}
        )
        assert result["success"] is False
        assert result["metadata"]["error"] == "configuration_error"

    @pytest.mark.asyncio
    async def test_missing_question(self, kosmos_adapter):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            result = await kosmos_adapter.run(task_config={}, input_files={})
            assert result["success"] is False
            assert result["metadata"]["error"] == "missing_parameter"

    @pytest.mark.asyncio
    async def test_successful_run(self, kosmos_adapter, tmp_path):
        mock_result = SubprocessResult(
            stdout="Research complete",
            stderr="",
            exit_code=0,
            timed_out=False
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("manyagents.adapters.kosmos_adapter.run_subprocess",
                       new_callable=AsyncMock, return_value=mock_result):
                os.chdir(tmp_path)
                result = await kosmos_adapter.run(
                    task_config={"research_question": "What is UMAP?"},
                    input_files={}
                )
                assert result["success"] is True
                assert "execution_time" in result["metadata"]

    @pytest.mark.asyncio
    async def test_timeout(self, kosmos_adapter, tmp_path):
        mock_result = SubprocessResult(
            stdout="partial",
            stderr="",
            exit_code=-1,
            timed_out=True
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("manyagents.adapters.kosmos_adapter.run_subprocess",
                       new_callable=AsyncMock, return_value=mock_result):
                os.chdir(tmp_path)
                result = await kosmos_adapter.run(
                    task_config={"research_question": "test", "timeout": 1},
                    input_files={}
                )
                assert result["success"] is False
                assert "timed out" in result["summary"].lower()

    def test_build_command_basic(self, kosmos_adapter):
        cmd = kosmos_adapter._build_command({"research_question": "test?"})
        assert "test?" in cmd
        assert "-m" in cmd
        assert "kosmos.cli" in cmd

    def test_build_command_with_domain(self, kosmos_adapter):
        cmd = kosmos_adapter._build_command({
            "research_question": "test?",
            "domain": "biology"
        })
        assert "--domain" in cmd
        assert "biology" in cmd
