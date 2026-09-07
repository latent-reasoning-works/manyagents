"""Tests for metric registry generation and lookup."""

import json
import pytest
from pathlib import Path
import tempfile
import shutil

from manyagents.adapters._generate_metric_registry import (
    scan_metric_configs,
    generate_metric_registry,
)
from manyagents.adapters.metric_registry import MetricRegistry, get_metric_registry


class TestMetricRegistryGeneration:
    """Test metric registry generation from manyLatents configs."""

    @pytest.mark.requires_manylatents
    def test_scan_metric_configs(self):
        """Test scanning manyLatents metric configs."""
        import manylatents
        metrics_dir = Path(manylatents.__file__).parent / 'configs' / 'metrics'

        registry = scan_metric_configs(metrics_dir)

        # Should find metrics
        assert len(registry) > 0, "No metrics found in manyLatents configs"

        # Check for known metrics
        known_metrics = [
            'participation_ratio',
            'local_intrinsic_dimensionality',
            'trustworthiness',
        ]

        found_metrics = [m for m in known_metrics if m in registry]
        assert len(found_metrics) > 0, f"None of {known_metrics} found in registry"

        # Check metric structure
        for metric_name, metric_info in registry.items():
            assert 'class' in metric_info, f"{metric_name} missing 'class'"
            assert 'group' in metric_info, f"{metric_name} missing 'group'"
            assert 'defaults' in metric_info, f"{metric_name} missing 'defaults'"
            assert 'partial' in metric_info, f"{metric_name} missing 'partial'"
            assert 'source_file' in metric_info, f"{metric_name} missing 'source_file'"

            # Check group is valid
            assert metric_info['group'] in ['embedding', 'dataset', 'module'], \
                f"{metric_name} has invalid group: {metric_info['group']}"

            # Check class path is valid format
            assert '.' in metric_info['class'], \
                f"{metric_name} class path invalid: {metric_info['class']}"

    @pytest.mark.requires_manylatents
    def test_generate_metric_registry(self):
        """Test full registry generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_registry.json'

            registry = generate_metric_registry(output_path, force=True)

            # Check file was created
            assert output_path.exists(), "Registry file not created"

            # Check structure
            assert '_metadata' in registry
            assert 'metrics' in registry

            # Check metadata
            metadata = registry['_metadata']
            assert 'generated_at' in metadata
            assert 'manylatents_version' in metadata
            assert 'metrics_scanned' in metadata
            assert 'algorithms_scanned' in metadata
            assert 'metric_groups' in metadata

            # Check metrics scanned
            assert metadata['metrics_scanned'] > 0, "No metrics scanned"
            assert metadata['algorithms_scanned'] > 0, "No algorithms scanned"

            # Check groups breakdown
            groups = metadata['metric_groups']
            assert 'embedding' in groups
            assert 'dataset' in groups
            assert 'module' in groups

            # Should have at least some metrics in each group
            assert groups['embedding'] > 0, "No embedding metrics found"

            # Validate JSON structure
            with open(output_path) as f:
                loaded = json.load(f)
                assert loaded == registry, "Saved JSON doesn't match in-memory registry"

    @pytest.mark.requires_manylatents
    def test_generate_registry_version_based_skip(self):
        """Test that registry skips regeneration if version matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_registry.json'

            # Generate first time
            registry1 = generate_metric_registry(output_path, force=True)
            mtime1 = output_path.stat().st_mtime

            # Wait a tiny bit
            import time
            time.sleep(0.01)

            # Generate again (should skip)
            registry2 = generate_metric_registry(output_path, force=False)
            mtime2 = output_path.stat().st_mtime

            # File should not have been regenerated (same mtime)
            assert mtime1 == mtime2, "Registry was regenerated despite matching version"

            # Registries should match
            assert registry1 == registry2

    @pytest.mark.requires_manylatents
    def test_generate_registry_force_regeneration(self):
        """Test force regeneration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_registry.json'

            # Generate first time
            registry1 = generate_metric_registry(output_path, force=True)
            mtime1 = output_path.stat().st_mtime

            # Wait a bit
            import time
            time.sleep(0.01)

            # Force regenerate
            registry2 = generate_metric_registry(output_path, force=True)
            mtime2 = output_path.stat().st_mtime

            # File should have been regenerated (different mtime)
            assert mtime2 > mtime1, "Registry was not regenerated with force=True"

            # Registries should still match (same content)
            assert registry1 == registry2


class TestMetricRegistry:
    """Test MetricRegistry class for querying and using the registry."""

    @pytest.fixture
    def temp_registry(self):
        """Create a temporary registry for testing."""
        tmpdir = tempfile.mkdtemp()
        registry_path = Path(tmpdir) / 'metric_registry.json'

        # Generate registry
        generate_metric_registry(registry_path, force=True)

        yield registry_path

        # Cleanup
        shutil.rmtree(tmpdir)

    @pytest.mark.requires_manylatents
    def test_load_registry(self, temp_registry):
        """Test loading registry."""
        registry = MetricRegistry(registry_path=temp_registry)

        assert len(registry) > 0, "Registry is empty"
        assert registry.metadata is not None
        assert 'manylatents_version' in registry.metadata

    @pytest.mark.requires_manylatents
    def test_list_metrics(self, temp_registry):
        """Test listing metrics."""
        registry = MetricRegistry(registry_path=temp_registry)

        # List all metrics
        all_metrics = registry.list_metrics()
        assert len(all_metrics) > 0

        # List by group
        embedding_metrics = registry.list_metrics(group='embedding')
        dataset_metrics = registry.list_metrics(group='dataset')
        module_metrics = registry.list_metrics(group='module')

        assert len(embedding_metrics) > 0, "No embedding metrics"
        assert len(dataset_metrics) >= 0  # May or may not have dataset metrics
        assert len(module_metrics) >= 0  # May or may not have module metrics

        # Total should match
        assert len(all_metrics) == len(embedding_metrics) + len(dataset_metrics) + len(module_metrics)

    @pytest.mark.requires_manylatents
    def test_get_metric_info(self, temp_registry):
        """Test getting metric info."""
        registry = MetricRegistry(registry_path=temp_registry)

        # Get info for known metric
        all_metrics = registry.list_metrics()
        assert len(all_metrics) > 0

        metric_name = all_metrics[0]
        info = registry.get_metric_info(metric_name)

        # Check structure
        assert 'class' in info
        assert 'group' in info
        assert 'defaults' in info
        assert 'partial' in info

        # Check types
        assert isinstance(info['class'], str)
        assert isinstance(info['group'], str)
        assert isinstance(info['defaults'], dict)
        assert isinstance(info['partial'], bool)

    @pytest.mark.requires_manylatents
    def test_get_metric_info_not_found(self, temp_registry):
        """Test getting info for non-existent metric."""
        registry = MetricRegistry(registry_path=temp_registry)

        with pytest.raises(KeyError, match="not found"):
            registry.get_metric_info('nonexistent_metric_xyz')

    @pytest.mark.requires_manylatents
    def test_get_metric_class(self, temp_registry):
        """Test getting metric class for instantiation."""
        registry = MetricRegistry(registry_path=temp_registry)

        # Get a metric that we know exists
        if 'participation_ratio' in registry:
            metric_class = registry.get_metric_class('participation_ratio')

            # Should be a class or callable (metrics can be classes or partials)
            assert callable(metric_class)

            # Should be from manylatents
            assert 'manylatents' in str(metric_class.__module__)

        elif len(registry) > 0:
            # Use first available metric
            metric_name = registry.list_metrics()[0]
            metric_class = registry.get_metric_class(metric_name)

            assert callable(metric_class)

    @pytest.mark.requires_manylatents
    def test_get_metric_class_caching(self, temp_registry):
        """Test that metric classes are cached."""
        registry = MetricRegistry(registry_path=temp_registry)

        if len(registry) > 0:
            metric_name = registry.list_metrics()[0]

            # Get class twice
            class1 = registry.get_metric_class(metric_name)
            class2 = registry.get_metric_class(metric_name)

            # Should be same object (cached)
            assert class1 is class2

    @pytest.mark.requires_manylatents
    def test_get_defaults(self, temp_registry):
        """Test getting default parameters."""
        registry = MetricRegistry(registry_path=temp_registry)

        if 'participation_ratio' in registry:
            defaults = registry.get_defaults('participation_ratio')

            assert isinstance(defaults, dict)
            # ParticipationRatio should have defaults
            assert len(defaults) > 0

    @pytest.mark.requires_manylatents
    def test_get_group(self, temp_registry):
        """Test getting metric group."""
        registry = MetricRegistry(registry_path=temp_registry)

        if 'participation_ratio' in registry:
            group = registry.get_group('participation_ratio')
            assert group == 'embedding'

    @pytest.mark.requires_manylatents
    def test_contains(self, temp_registry):
        """Test __contains__ method."""
        registry = MetricRegistry(registry_path=temp_registry)

        if len(registry) > 0:
            metric_name = registry.list_metrics()[0]
            assert metric_name in registry
            assert 'nonexistent_metric_xyz' not in registry

    @pytest.mark.requires_manylatents
    def test_repr(self, temp_registry):
        """Test string representation."""
        registry = MetricRegistry(registry_path=temp_registry)

        repr_str = repr(registry)
        assert 'MetricRegistry' in repr_str
        assert 'metrics=' in repr_str

    @pytest.mark.requires_manylatents
    def test_singleton_pattern(self):
        """Test get_metric_registry singleton."""
        # This test requires a real registry to exist
        # We'll skip if it doesn't exist
        try:
            registry1 = get_metric_registry()
            registry2 = get_metric_registry()

            # Should be same instance
            assert registry1 is registry2
        except FileNotFoundError:
            pytest.skip("Real metric registry not generated yet")


class TestMetricRegistryIntegration:
    """Integration tests with real manyLatents metrics."""

    @pytest.fixture
    def temp_registry(self):
        """Create a temporary registry for testing."""
        tmpdir = tempfile.mkdtemp()
        registry_path = Path(tmpdir) / 'metric_registry.json'
        generate_metric_registry(registry_path, force=True)
        yield registry_path
        shutil.rmtree(tmpdir)

    @pytest.mark.requires_manylatents
    def test_registry_has_sufficient_metrics(self, temp_registry):
        """Test that registry has at least 25 metrics (success criterion)."""
        registry = MetricRegistry(registry_path=temp_registry)

        metrics_count = len(registry)
        assert metrics_count >= 25, \
            f"Registry should have at least 25 metrics, found {metrics_count}"

    @pytest.mark.requires_manylatents
    def test_known_metrics_present(self, temp_registry):
        """Test that known important metrics are present."""
        registry = MetricRegistry(registry_path=temp_registry)

        # These metrics should definitely exist
        important_metrics = [
            'participation_ratio',
            'local_intrinsic_dimensionality',
            'trustworthiness',
        ]

        for metric in important_metrics:
            if metric not in registry:
                # Try to find it with different names
                all_metrics = registry.list_metrics()
                pytest.fail(
                    f"Important metric '{metric}' not found. "
                    f"Available: {', '.join(all_metrics[:10])}..."
                )

    @pytest.mark.requires_manylatents
    def test_metric_instantiation(self, temp_registry):
        """Test that we can actually get metric classes from registry."""
        registry = MetricRegistry(registry_path=temp_registry)

        # Try to get a known metric class
        if 'participation_ratio' in registry:
            metric_class = registry.get_metric_class('participation_ratio')
            defaults = registry.get_defaults('participation_ratio')

            # The class should be importable and callable
            assert callable(metric_class)

            # Defaults should be a dict
            assert isinstance(defaults, dict)

            # Should have expected defaults for participation_ratio
            assert 'n_neighbors' in defaults or 'return_per_sample' in defaults
