#!/usr/bin/env python3
"""
Memory profiling script for VASA training.
Monitors CPU and GPU memory usage during training.
"""

import torch
import psutil
import os
import time
from contextlib import contextmanager
import gc

class MemoryProfiler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @contextmanager
    def profile(self, name="Operation"):
        """Context manager to profile memory usage"""
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        # Get initial memory stats
        cpu_mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_reserved_before = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        else:
            gpu_mem_before = 0
            gpu_reserved_before = 0

        start_time = time.time()

        try:
            yield
        finally:
            # Get final memory stats
            cpu_mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
            if torch.cuda.is_available():
                gpu_mem_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_reserved_after = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            else:
                gpu_mem_after = 0
                gpu_reserved_after = 0

            elapsed_time = time.time() - start_time

            # Calculate deltas
            cpu_delta = cpu_mem_after - cpu_mem_before
            gpu_delta = gpu_mem_after - gpu_mem_before
            gpu_reserved_delta = gpu_reserved_after - gpu_reserved_before

            # Print report
            print(f"\n{'='*60}")
            print(f"Memory Profile: {name}")
            print(f"{'='*60}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"\nCPU Memory:")
            print(f"  Before: {cpu_mem_before:.1f} MB")
            print(f"  After:  {cpu_mem_after:.1f} MB")
            print(f"  Delta:  {cpu_delta:+.1f} MB")

            if torch.cuda.is_available():
                print(f"\nGPU Memory (Allocated):")
                print(f"  Before: {gpu_mem_before:.1f} MB")
                print(f"  After:  {gpu_mem_after:.1f} MB")
                print(f"  Delta:  {gpu_delta:+.1f} MB")

                print(f"\nGPU Memory (Reserved):")
                print(f"  Before: {gpu_reserved_before:.1f} MB")
                print(f"  After:  {gpu_reserved_after:.1f} MB")
                print(f"  Delta:  {gpu_reserved_delta:+.1f} MB")

                # Show peak memory usage
                peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"\nPeak GPU Memory: {peak_gpu:.1f} MB")
            print(f"{'='*60}\n")


def test_data_loading():
    """Test memory usage during data loading"""
    from vasa_dataset import VASAIntegratedDataset
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from vasa_sampler import WindowSequenceSampler, create_window_sequence_collate_fn

    profiler = MemoryProfiler()

    # Load config
    config = OmegaConf.load('overfit_config.yaml')

    with profiler.profile("Dataset Creation"):
        dataset = VASAIntegratedDataset(
            video_dir=config.paths.videos_train,
            cache_type='single_bucket',
            config=config,
            split='train'
        )

    with profiler.profile("Sampler Creation"):
        sampler = WindowSequenceSampler(
            dataset=dataset,
            batch_size=2,
            sequence_length=4,
            shuffle=False,
            drop_last=False
        )

        collate_fn = create_window_sequence_collate_fn(
            context_size=config.motion.context_size if hasattr(config, 'motion') else 10
        )

    with profiler.profile("DataLoader Creation"):
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    # Test loading a few batches
    with profiler.profile("Loading 3 Batches"):
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break

            with profiler.profile(f"Batch {i} Processing"):
                if batch is not None:
                    # Simulate moving data to GPU
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            _ = value.to(device, non_blocking=True)

                    # Force synchronization
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()

    print("\nMemory profiling complete!")


def compare_optimizations():
    """Compare memory usage with and without optimizations"""
    import sys

    print("Testing memory usage with optimizations...")
    print("=" * 80)

    # Test with current optimized settings
    print("\n1. WITH OPTIMIZATIONS (pin_memory=True, num_workers=2)")
    print("-" * 40)
    test_data_loading()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    # Temporarily modify settings to test without optimizations
    print("\n2. WITHOUT OPTIMIZATIONS (pin_memory=False, num_workers=0)")
    print("-" * 40)

    # Monkey-patch to disable optimizations
    original_dataloader = torch.utils.data.DataLoader

    class UnoptimizedDataLoader(original_dataloader):
        def __init__(self, *args, **kwargs):
            kwargs['pin_memory'] = False
            kwargs['num_workers'] = 0
            kwargs['persistent_workers'] = False
            kwargs.pop('prefetch_factor', None)
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = UnoptimizedDataLoader

    try:
        test_data_loading()
    finally:
        # Restore original DataLoader
        torch.utils.data.DataLoader = original_dataloader

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("\nKey observations:")
    print("- pin_memory=True reduces CPU-GPU transfer time")
    print("- num_workers>0 allows parallel data loading")
    print("- persistent_workers=True avoids worker recreation overhead")
    print("- Moving tensors to GPU early reduces CPU memory pressure")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory profiling for VASA training")
    parser.add_argument('--compare', action='store_true',
                       help='Compare optimized vs unoptimized settings')
    args = parser.parse_args()

    if args.compare:
        compare_optimizations()
    else:
        test_data_loading()