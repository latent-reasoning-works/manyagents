"""Tests for cached metrics setup in ManyLatentsAdapter."""

import pytest
import numpy as np
from manyagents.adapters.manylatents_adapter import ManyLatentsAdapter


class TestMetricSpecParsing:
    """Test parsing of metric specifications."""

    def test_parse_simple_string(self):
        """Test parsing simple metric name string."""
        adapter = ManyLatentsAdapter()
        name, params = adapter._parse_metric_spec('participation_ratio')

        assert name == 'participation_ratio'
        assert params == {}

    def test_parse_dict_with_params(self):
        """Test parsing dict with parameters."""
        adapter = ManyLatentsAdapter()
        name, params = adapter._parse_metric_spec({'lid': {'k': 30}})

        assert name == 'lid'
        assert params == {'k': 30}

    def test_parse_dict_with_empty_params(self):
        """Test parsing dict with empty params."""
        adapter = ManyLatentsAdapter()
        name, params = adapter._parse_metric_spec({'participation_ratio': {}})

        assert name == 'participation_ratio'
        assert params == {}

    def test_parse_invalid_multi_key_dict(self):
        """Test that multi-key dict raises error."""
        adapter = ManyLatentsAdapter()

        with pytest.raises(ValueError, match="exactly one key"):
            adapter._parse_metric_spec({'lid': {'k': 30}, 'pr': {}})

    def test_parse_invalid_non_dict_params(self):
        """Test that non-dict params raise error."""
        adapter = ManyLatentsAdapter()

        with pytest.raises(ValueError, match="must be a dict"):
            adapter._parse_metric_spec({'lid': 30})  # Should be {'lid': {'k': 30}}

    def test_parse_invalid_type(self):
        """Test that invalid types raise error."""
        adapter = ManyLatentsAdapter()

        with pytest.raises(ValueError, match="Invalid metric spec type"):
            adapter._parse_metric_spec(123)

        with pytest.raises(ValueError, match="Invalid metric spec type"):
            adapter._parse_metric_spec(['lid'])


class TestMetricSetup:
    """Test metric setup and caching functionality."""

    @pytest.mark.requires_manylatents
    def test_setup_simple_metrics(self):
        """Test setting up metrics with simple names."""
        adapter = ManyLatentsAdapter()

        # Setup metrics
        adapter.setup_metrics(['participation_ratio', 'local_intrinsic_dimensionality'])

        # Check cache was populated
        assert 'participation_ratio' in adapter._metric_cache
        assert 'local_intrinsic_dimensionality' in adapter._metric_cache

        # Check cached mode enabled
        assert adapter._cached_mode is True

        # Check metric objects
        pr_cache = adapter._metric_cache['participation_ratio']
        assert 'object' in pr_cache
        assert 'group' in pr_cache
        assert 'class' in pr_cache
        assert 'params' in pr_cache

        # Check group
        assert pr_cache['group'] == 'embedding'

    @pytest.mark.requires_manylatents
    def test_setup_with_overrides(self):
        """Test setting up metrics with parameter overrides."""
        adapter = ManyLatentsAdapter()

        # Setup with override
        adapter.setup_metrics([{'local_intrinsic_dimensionality': {'k': 30}}])

        assert 'local_intrinsic_dimensionality' in adapter._metric_cache

        # Check params were applied
        lid_cache = adapter._metric_cache['local_intrinsic_dimensionality']
        assert lid_cache['params']['k'] == 30

    @pytest.mark.requires_manylatents
    def test_setup_with_global_overrides(self):
        """Test global parameter overrides."""
        adapter = ManyLatentsAdapter()

        # Setup with global override
        adapter.setup_metrics(
            ['participation_ratio'],
            return_per_sample=False  # Override default
        )

        pr_cache = adapter._metric_cache['participation_ratio']
        assert pr_cache['params']['return_per_sample'] is False

    @pytest.mark.requires_manylatents
    def test_setup_mixed_specs(self):
        """Test mixing simple names and override dicts."""
        adapter = ManyLatentsAdapter()

        adapter.setup_metrics([
            'participation_ratio',
            {'local_intrinsic_dimensionality': {'k': 30}},
            'trustworthiness'
        ])

        assert len(adapter._metric_cache) == 3
        assert 'participation_ratio' in adapter._metric_cache
        assert 'local_intrinsic_dimensionality' in adapter._metric_cache
        assert 'trustworthiness' in adapter._metric_cache

        # Check override was applied
        lid_cache = adapter._metric_cache['local_intrinsic_dimensionality']
        assert lid_cache['params']['k'] == 30

    @pytest.mark.requires_manylatents
    def test_setup_unknown_metric(self):
        """Test that unknown metric raises error."""
        adapter = ManyLatentsAdapter()

        with pytest.raises(KeyError, match="not found"):
            adapter.setup_metrics(['nonexistent_metric_xyz'])

    @pytest.mark.requires_manylatents
    def test_registry_lazy_loading(self):
        """Test that registry is lazy-loaded."""
        adapter = ManyLatentsAdapter()

        # Registry should not be loaded yet
        assert adapter._metric_registry is None

        # Setup metrics (triggers loading)
        adapter.setup_metrics(['participation_ratio'])

        # Now registry should be loaded
        assert adapter._metric_registry is not None

    @pytest.mark.requires_manylatents
    def test_multiple_setup_calls(self):
        """Test calling setup_metrics multiple times."""
        adapter = ManyLatentsAdapter()

        # First setup
        adapter.setup_metrics(['participation_ratio'])
        assert len(adapter._metric_cache) == 1

        # Second setup (should add to cache)
        adapter.setup_metrics(['local_intrinsic_dimensionality'])
        assert len(adapter._metric_cache) == 2

        # Both metrics should be cached
        assert 'participation_ratio' in adapter._metric_cache
        assert 'local_intrinsic_dimensionality' in adapter._metric_cache


class TestCachedMetricCallability:
    """Test that cached metrics are callable and work correctly."""

    @pytest.mark.requires_manylatents
    def test_cached_metrics_are_callable(self):
        """Test that all cached metrics are callable."""
        adapter = ManyLatentsAdapter()

        adapter.setup_metrics([
            'participation_ratio',
            'local_intrinsic_dimensionality',
            'trustworthiness'
        ])

        for metric_name, metric_cache in adapter._metric_cache.items():
            metric_obj = metric_cache['object']
            assert callable(metric_obj), f"Metric '{metric_name}' is not callable"

    @pytest.mark.requires_manylatents
    def test_metric_cache_structure(self):
        """Test that metric cache has expected structure."""
        adapter = ManyLatentsAdapter()

        adapter.setup_metrics(['participation_ratio'])

        pr_cache = adapter._metric_cache['participation_ratio']

        # Check all expected keys exist
        assert 'object' in pr_cache
        assert 'group' in pr_cache
        assert 'class' in pr_cache
        assert 'params' in pr_cache

        # Check types
        assert callable(pr_cache['object'])
        assert isinstance(pr_cache['group'], str)
        assert isinstance(pr_cache['class'], str)
        assert isinstance(pr_cache['params'], dict)

        # Check group is valid
        assert pr_cache['group'] in ['embedding', 'dataset', 'module']

    @pytest.mark.requires_manylatents
    def test_parameter_priority(self):
        """Test parameter merge priority: defaults < global < specific."""
        adapter = ManyLatentsAdapter()

        # Get default value for n_neighbors
        from manyagents.adapters.metric_registry import MetricRegistry
        registry = MetricRegistry()
        defaults = registry.get_defaults('participation_ratio')
        default_n_neighbors = defaults.get('n_neighbors', 25)

        # Setup with both global and specific overrides
        adapter.setup_metrics(
            [
                'trustworthiness',  # Uses global override
                {'participation_ratio': {'n_neighbors': 50}}  # Specific override
            ],
            n_neighbors=30  # Global override
        )

        # Trustworthiness should use global override
        tw_cache = adapter._metric_cache['trustworthiness']
        assert tw_cache['params']['n_neighbors'] == 30

        # ParticipationRatio should use specific override (highest priority)
        pr_cache = adapter._metric_cache['participation_ratio']
        assert pr_cache['params']['n_neighbors'] == 50


class TestMetricSetupIntegration:
    """Integration tests with actual metric registry."""

    @pytest.mark.requires_manylatents
    def test_setup_with_real_metrics(self):
        """Test setup with actual metrics from registry."""
        adapter = ManyLatentsAdapter()

        # Get some real metrics
        from manyagents.adapters.metric_registry import MetricRegistry
        registry = MetricRegistry()

        # Get first 3 embedding metrics
        embedding_metrics = registry.list_metrics(group='embedding')[:3]

        # Should successfully set them up
        adapter.setup_metrics(embedding_metrics)

        assert len(adapter._metric_cache) == len(embedding_metrics)
        for metric_name in embedding_metrics:
            assert metric_name in adapter._metric_cache

    @pytest.mark.requires_manylatents
    def test_metric_groups_preserved(self):
        """Test that metric groups are preserved correctly."""
        adapter = ManyLatentsAdapter()

        from manyagents.adapters.metric_registry import MetricRegistry
        registry = MetricRegistry()

        # Get metrics from each group
        embedding_metric = registry.list_metrics(group='embedding')[0]
        dataset_metrics = registry.list_metrics(group='dataset')
        module_metrics = registry.list_metrics(group='module')

        setup_metrics = [embedding_metric]
        if dataset_metrics:
            setup_metrics.append(dataset_metrics[0])
        if module_metrics:
            setup_metrics.append(module_metrics[0])

        adapter.setup_metrics(setup_metrics)

        # Check groups are preserved
        assert adapter._metric_cache[embedding_metric]['group'] == 'embedding'
        if dataset_metrics:
            assert adapter._metric_cache[dataset_metrics[0]]['group'] == 'dataset'
        if module_metrics:
            assert adapter._metric_cache[module_metrics[0]]['group'] == 'module'

    @pytest.mark.requires_manylatents
    def test_empty_metric_list(self):
        """Test setup with empty metric list."""
        adapter = ManyLatentsAdapter()

        adapter.setup_metrics([])

        assert adapter._metric_cache == {}
        assert adapter._cached_mode is True  # Still enabled

    @pytest.mark.requires_manylatents
    def test_duplicate_metrics(self):
        """Test that duplicate metrics in list are handled."""
        adapter = ManyLatentsAdapter()

        # Setup same metric twice
        adapter.setup_metrics([
            'participation_ratio',
            'participation_ratio'  # Duplicate
        ])

        # Should only be cached once (last wins)
        assert len(adapter._metric_cache) == 1
        assert 'participation_ratio' in adapter._metric_cache
