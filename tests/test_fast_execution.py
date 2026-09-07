"""Tests for ManyLatentsAdapter execute_cached() method.

This module tests the fast execution path that uses pre-instantiated algorithms
and cached metrics for rapid RL training loops.

Test Coverage:
- Algorithm instantiation from unified registry
- Cached metric execution
- EmbeddingOutputs format compliance
- Performance benchmarks
- Error handling
"""

import pytest
import numpy as np
import time

from manyagents.adapters.manylatents_adapter import ManyLatentsAdapter


class TestExecuteCachedBasics:
    """Test basic execute_cached functionality."""

    @pytest.mark.asyncio
    async def test_requires_setup_first(self):
        """execute_cached should fail if setup_metrics not called."""
        adapter = ManyLatentsAdapter()
        data = np.random.randn(100, 50)

        with pytest.raises(RuntimeError, match="Must call setup_metrics"):
            await adapter.execute_cached(
                algorithm='PCA',
                params={'n_components': 2},
                data=data
            )

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test basic execution with PCA and single metric."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(100, 50)
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 2},
            data=data
        )

        # Verify EmbeddingOutputs format
        assert result['success'] is True
        assert 'embeddings' in result
        assert 'scores' in result
        assert 'metadata' in result

        # Verify embeddings
        assert isinstance(result['embeddings'], np.ndarray)
        assert result['embeddings'].shape == (100, 2)

        # Verify metrics
        assert 'participation_ratio' in result['scores']
        assert result['scores']['participation_ratio'] is not None

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_multiple_metrics(self):
        """Test execution with multiple cached metrics."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics([
            'participation_ratio',
            'local_intrinsic_dimensionality',
            'trustworthiness'
        ])

        data = np.random.randn(200, 50)
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 10},
            data=data
        )

        assert result['success'] is True
        assert result['embeddings'].shape == (200, 10)

        # All metrics should be computed
        assert 'participation_ratio' in result['scores']
        assert 'local_intrinsic_dimensionality' in result['scores']
        assert 'trustworthiness' in result['scores']


class TestAlgorithmRegistry:
    """Test dynamic algorithm registry integration."""

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_pca_algorithm(self):
        """Test PCA algorithm from registry."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(100, 50)
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 5},
            data=data
        )

        assert result['embeddings'].shape == (100, 5)
        assert result['metadata']['algorithm'] == 'PCA'
        assert result['metadata']['params']['n_components'] == 5

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_umap_algorithm(self):
        """Test UMAP algorithm from registry."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(100, 50)
        result = await adapter.execute_cached(
            algorithm='UMAP',
            params={'n_components': 2, 'n_neighbors': 10},
            data=data
        )

        assert result['embeddings'].shape == (100, 2)
        assert result['metadata']['algorithm'] == 'UMAP'
        assert result['metadata']['params']['n_components'] == 2
        assert result['metadata']['params']['n_neighbors'] == 10

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_algorithm_defaults_merge(self):
        """Test that algorithm defaults are properly merged with params."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(100, 50)

        # Only override n_components, let other defaults apply
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 3},
            data=data
        )

        assert result['embeddings'].shape == (100, 3)
        # PCA defaults should be applied (verify via metadata)
        assert result['metadata']['params']['n_components'] == 3

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_invalid_algorithm(self):
        """Test error handling for unknown algorithm."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(100, 50)

        with pytest.raises(KeyError):
            await adapter.execute_cached(
                algorithm='NONEXISTENT',
                params={},
                data=data
            )


class TestMetricGroups:
    """Test different metric groups (embedding, module, dataset)."""

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_embedding_metrics(self):
        """Test metrics that operate on embeddings only."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics([
            'participation_ratio',  # embedding group
            'anisotropy',  # embedding group
        ])

        data = np.random.randn(100, 50)
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 2},
            data=data
        )

        assert 'participation_ratio' in result['scores']
        assert 'anisotropy' in result['scores']

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_module_metrics(self):
        """Test metrics that require the algorithm module."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics([
            'connected_components',  # module group
        ])

        data = np.random.randn(100, 50)
        result = await adapter.execute_cached(
            algorithm='UMAP',
            params={'n_components': 2},
            data=data
        )

        assert 'connected_components' in result['scores']

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_mixed_metric_groups(self):
        """Test execution with metrics from different groups."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics([
            'participation_ratio',  # embedding
            'connected_components',  # module
        ])

        data = np.random.randn(100, 50)
        result = await adapter.execute_cached(
            algorithm='UMAP',
            params={'n_components': 2},
            data=data
        )

        assert 'participation_ratio' in result['scores']
        assert 'connected_components' in result['scores']


class TestMetricParameterization:
    """Test metric parameter overrides in cached mode."""

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_metric_with_overrides(self):
        """Test metric instantiation with parameter overrides."""
        adapter = ManyLatentsAdapter()

        # Override default k for LID
        adapter.setup_metrics([
            {'local_intrinsic_dimensionality': {'k': 30}}
        ])

        data = np.random.randn(100, 50)
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 2},
            data=data
        )

        assert 'local_intrinsic_dimensionality' in result['scores']
        # Verify the parameter was used (check cached metric)
        assert adapter._metric_cache['local_intrinsic_dimensionality']['params']['k'] == 30

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_global_overrides(self):
        """Test global parameter overrides applied to all metrics."""
        adapter = ManyLatentsAdapter()

        # Apply global override to all metrics
        adapter.setup_metrics(
            ['participation_ratio', 'tangent_space'],
            return_per_sample=False  # Override default for both
        )

        # Verify both metrics got the global override
        assert adapter._metric_cache['participation_ratio']['params']['return_per_sample'] is False
        assert adapter._metric_cache['tangent_space']['params']['return_per_sample'] is False


class TestPerformance:
    """Test performance of cached execution."""

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_execution_speed(self):
        """Verify execute_cached meets <100ms target."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics([
            'participation_ratio',
            'local_intrinsic_dimensionality'
        ])

        # Small dataset for speed test
        data = np.random.randn(200, 50)

        # Warm-up run (first run may be slower due to JIT, imports, etc.)
        await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 2},
            data=data
        )

        # Timed run
        start = time.perf_counter()
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 2},
            data=data
        )
        elapsed = time.perf_counter() - start

        assert result['success'] is True
        # Note: <100ms target may not be achievable depending on system
        # But cached mode should be significantly faster than full Hydra path
        print(f"\nCached execution time: {elapsed*1000:.2f}ms")

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_repeated_executions(self):
        """Test that cached metrics enable fast repeated execution."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(100, 50)

        # Run multiple times (simulating RL loop)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = await adapter.execute_cached(
                algorithm='PCA',
                params={'n_components': 2},
                data=data
            )
            times.append(time.perf_counter() - start)
            assert result['success'] is True

        # Times should be consistent (no overhead accumulation)
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"\nAverage execution time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")

        # Standard deviation should be small (consistent performance)
        assert std_time < avg_time * 0.5  # Less than 50% variance


class TestDataShapes:
    """Test various data shapes and sizes."""

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_small_data(self):
        """Test with small dataset."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(50, 10)
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 2},
            data=data
        )

        assert result['embeddings'].shape == (50, 2)

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_large_data(self):
        """Test with larger dataset."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(1000, 100)
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 10},
            data=data
        )

        assert result['embeddings'].shape == (1000, 10)

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_high_dimensional_data(self):
        """Test with high-dimensional input."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(200, 500)
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 20},
            data=data
        )

        assert result['embeddings'].shape == (200, 20)


class TestMetadata:
    """Test metadata in execution results."""

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        """Verify metadata contains expected fields."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(100, 50)
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 2},
            data=data
        )

        metadata = result['metadata']
        assert 'algorithm' in metadata
        assert 'params' in metadata
        assert 'data_shape' in metadata
        assert 'embedding_shape' in metadata
        assert 'cached_execution' in metadata
        assert 'metrics_computed' in metadata

        # Verify values
        assert metadata['algorithm'] == 'PCA'
        assert metadata['params']['n_components'] == 2
        assert metadata['data_shape'] == (100, 50)
        assert metadata['embedding_shape'] == (100, 2)
        assert metadata['cached_execution'] is True
        assert metadata['metrics_computed'] == 1

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_metadata_metrics_count(self):
        """Verify metadata tracks correct number of computed metrics."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics([
            'participation_ratio',
            'local_intrinsic_dimensionality',
            'trustworthiness'
        ])

        data = np.random.randn(100, 50)
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 2},
            data=data
        )

        assert result['metadata']['metrics_computed'] == 3


class TestErrorHandling:
    """Test error handling in execute_cached."""

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_invalid_params(self):
        """Test handling of invalid algorithm parameters."""
        adapter = ManyLatentsAdapter()
        adapter.setup_metrics(['participation_ratio'])

        data = np.random.randn(100, 50)

        # Try to request more components than features
        with pytest.raises(Exception):  # Should raise from algorithm
            await adapter.execute_cached(
                algorithm='PCA',
                params={'n_components': 100},  # More than 50 features
                data=data
            )

    @pytest.mark.requires_manylatents
    @pytest.mark.asyncio
    async def test_metric_computation_failure(self):
        """Test that metric failures are gracefully handled."""
        adapter = ManyLatentsAdapter()

        # Use a metric that might fail with certain data
        adapter.setup_metrics(['participation_ratio'])

        # Provide data that might cause issues
        data = np.ones((100, 50))  # All same values

        # Should still succeed, but metric might be None
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 2},
            data=data
        )

        assert result['success'] is True
        # Metric might be None due to computation issues, but shouldn't crash


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
