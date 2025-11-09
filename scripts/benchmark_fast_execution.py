"""Benchmark fast execution path vs normal Hydra path.

This script compares the performance of:
1. Normal path: adapter.run() with Hydra config
2. Fast path: adapter.execute_cached() with pre-instantiated metrics

Expected results:
- Normal path: ~500-1000ms per execution (Hydra overhead)
- Fast path: ~50-100ms per execution (no Hydra overhead)
- Speedup: 5-10x faster
"""

import asyncio
import time
import numpy as np
from pathlib import Path

from manyagents.adapters.manylatents_adapter import ManyLatentsAdapter


async def benchmark_normal_path(n_trials: int = 5):
    """Benchmark normal execution path with Hydra."""
    print("\n" + "="*60)
    print("Benchmarking NORMAL path (with Hydra overhead)")
    print("="*60)

    adapter = ManyLatentsAdapter()
    data = np.random.randn(200, 50)

    times = []
    for i in range(n_trials):
        start = time.perf_counter()

        # Normal path using adapter.run()
        result = await adapter.run(
            task_config={
                'algorithm': 'PCA',
                'data': 'swissroll',  # Will be overridden by input_data
                'n_components': 2,
            },
            input_files={},
            input_data=data
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"  Trial {i+1}/{n_trials}: {elapsed*1000:.2f}ms")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n  Average: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"  Min: {min(times)*1000:.2f}ms")
    print(f"  Max: {max(times)*1000:.2f}ms")

    return avg_time, std_time


async def benchmark_fast_path(n_trials: int = 50):
    """Benchmark fast execution path with cached metrics."""
    print("\n" + "="*60)
    print("Benchmarking FAST path (cached metrics)")
    print("="*60)

    adapter = ManyLatentsAdapter()

    # Setup phase (only once)
    print("\n  Setup phase (one-time cost)...")
    setup_start = time.perf_counter()
    adapter.setup_metrics([
        'participation_ratio',
        'local_intrinsic_dimensionality',
        'trustworthiness'
    ])
    setup_time = time.perf_counter() - setup_start
    print(f"  Setup time: {setup_time*1000:.2f}ms")

    # Execution phase (repeated many times)
    print(f"\n  Execution phase ({n_trials} trials)...")
    data = np.random.randn(200, 50)

    times = []
    for i in range(n_trials):
        start = time.perf_counter()

        # Fast path using execute_cached()
        result = await adapter.execute_cached(
            algorithm='PCA',
            params={'n_components': 2},
            data=data
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if i < 5 or i >= n_trials - 5:
            print(f"  Trial {i+1}/{n_trials}: {elapsed*1000:.2f}ms")
        elif i == 5:
            print(f"  ... (trials 6-{n_trials-5}) ...")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n  Average: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"  Min: {min(times)*1000:.2f}ms")
    print(f"  Max: {max(times)*1000:.2f}ms")

    return avg_time, std_time, setup_time


async def main():
    """Run benchmarks and compare results."""
    print("\n" + "="*60)
    print("FAST EXECUTION BENCHMARK")
    print("="*60)
    print("\nThis benchmark compares:")
    print("  1. Normal path: adapter.run() with Hydra config")
    print("  2. Fast path: adapter.execute_cached() with cached metrics")
    print("\nConfiguration:")
    print("  - Algorithm: PCA")
    print("  - Data shape: (200, 50)")
    print("  - Target components: 2")
    print("  - Metrics: participation_ratio, lid, trustworthiness")

    # Benchmark normal path
    normal_avg, normal_std = await benchmark_normal_path(n_trials=5)

    # Benchmark fast path
    fast_avg, fast_std, setup_time = await benchmark_fast_path(n_trials=50)

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    speedup = normal_avg / fast_avg

    print(f"\n  Normal path:  {normal_avg*1000:.2f}ms ± {normal_std*1000:.2f}ms")
    print(f"  Fast path:    {fast_avg*1000:.2f}ms ± {fast_std*1000:.2f}ms")
    print(f"  Setup cost:   {setup_time*1000:.2f}ms (one-time)")
    print(f"\n  Speedup:      {speedup:.1f}x faster")

    # Break-even analysis
    breakeven = setup_time / (normal_avg - fast_avg)
    print(f"\n  Break-even:   {int(breakeven)} executions")
    print(f"                (after {int(breakeven)} runs, fast path saves time)")

    # RL training projection
    episodes_per_run = 1000
    normal_total = normal_avg * episodes_per_run
    fast_total = setup_time + (fast_avg * episodes_per_run)
    time_saved = normal_total - fast_total

    print(f"\n  For {episodes_per_run} RL episodes:")
    print(f"    Normal path total:  {normal_total:.1f}s ({normal_total/60:.1f}min)")
    print(f"    Fast path total:    {fast_total:.1f}s ({fast_total/60:.1f}min)")
    print(f"    Time saved:         {time_saved:.1f}s ({time_saved/60:.1f}min)")
    print(f"    Efficiency gain:    {(time_saved/normal_total)*100:.1f}%")

    print("\n" + "="*60)


if __name__ == '__main__':
    asyncio.run(main())
