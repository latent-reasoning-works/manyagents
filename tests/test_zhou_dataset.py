"""Tests for Zhou et al. dataset loader."""
import pytest


class TestParseSteps:

    def test_strip_step_marker(self):
        from manyagents.datasets.zhou_reasoning_flow import _strip_step_marker

        assert _strip_step_marker("[1] A→B") == "A→B"
        assert _strip_step_marker("[12] Long step text") == "Long step text"
        assert _strip_step_marker("No marker") == "No marker"

    def test_parse_topic_field(self):
        from manyagents.datasets.zhou_reasoning_flow import _parse_topic_field

        assert _parse_topic_field("weather_en") == ("weather", "en")
        assert _parse_topic_field("network_security_zh") == ("network_security", "zh")
        assert _parse_topic_field("finance_de") == ("finance", "de")
        assert _parse_topic_field("astronomy_ja") == ("astronomy", "ja")
        # Unknown language suffix
        assert _parse_topic_field("something_xx") == ("something_xx", "unknown")


class TestLoadDataset:

    def test_load_returns_list_of_dicts(self):
        """Smoke test — can we load anything at all?"""
        from manyagents.datasets.zhou_reasoning_flow import load_zhou_reasoning_flow

        try:
            entries = load_zhou_reasoning_flow(n_samples=5)
        except Exception as e:
            pytest.skip(f"Dataset not accessible: {e}")

        assert isinstance(entries, list)
        assert len(entries) <= 5

        if entries:
            entry = entries[0]
            assert "task_info" in entry
            assert "steps" in entry
            assert "logic_type" in entry
            assert "topic" in entry
            assert "language" in entry
            assert isinstance(entry["steps"], list)
            assert len(entry["steps"]) > 0

    def test_filter_by_language(self):
        """Language filter works."""
        from manyagents.datasets.zhou_reasoning_flow import load_zhou_reasoning_flow

        try:
            entries = load_zhou_reasoning_flow(n_samples=50, languages=["en"])
        except Exception as e:
            pytest.skip(f"Dataset not accessible: {e}")

        for entry in entries:
            assert entry["language"] == "en"

    def test_filter_by_logic_type(self):
        """Logic type filter works."""
        from manyagents.datasets.zhou_reasoning_flow import load_zhou_reasoning_flow

        try:
            entries = load_zhou_reasoning_flow(logic_types=["logicA"])
        except Exception as e:
            pytest.skip(f"Dataset not accessible: {e}")

        assert len(entries) > 0
        for entry in entries:
            assert entry["logic_type"] == "logicA"

    def test_filter_by_topic(self):
        """Topic filter works."""
        from manyagents.datasets.zhou_reasoning_flow import load_zhou_reasoning_flow

        try:
            entries = load_zhou_reasoning_flow(topics=["weather"], n_samples=20)
        except Exception as e:
            pytest.skip(f"Dataset not accessible: {e}")

        for entry in entries:
            assert entry["topic"] == "weather"

    def test_exclude_symbolic(self):
        """Can exclude symbolic-only entries."""
        from manyagents.datasets.zhou_reasoning_flow import load_zhou_reasoning_flow

        try:
            entries = load_zhou_reasoning_flow(
                logic_types=["logicA"], include_symbolic=False
            )
        except Exception as e:
            pytest.skip(f"Dataset not accessible: {e}")

        for entry in entries:
            assert entry["topic"] != "symbolic"

    def test_steps_are_marker_stripped(self):
        """Steps have [N] markers stripped by default."""
        from manyagents.datasets.zhou_reasoning_flow import load_zhou_reasoning_flow

        try:
            entries = load_zhou_reasoning_flow(n_samples=1)
        except Exception as e:
            pytest.skip(f"Dataset not accessible: {e}")

        if entries:
            for step in entries[0]["steps"]:
                assert not step.startswith("[")

    def test_total_entries(self):
        """Full dataset has 2430 entries."""
        from manyagents.datasets.zhou_reasoning_flow import load_zhou_reasoning_flow

        try:
            entries = load_zhou_reasoning_flow()
        except Exception as e:
            pytest.skip(f"Dataset not accessible: {e}")

        assert len(entries) == 2430
