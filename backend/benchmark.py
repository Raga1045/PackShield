"""Inference latency and throughput benchmarking module for DefektGuard.

Measures:
- Model loading / cold start time
- Warm-up execution
- Single-sample latency percentiles (Mean, Median, p95, p99, Std Dev)
- Batch throughput scaling (Batch sizes 1, 4, 8, 16, 32)
- Hardware & software execution environment documentation
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config


def get_system_metadata() -> Dict[str, str]:
    """Captures exact hardware and execution environment details."""
    devices = tf.config.list_physical_devices()
    device_names = [f"{d.device_type}:{d.name}" for d in devices]

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "devices": device_names,
        "input_resolution": f"{config.IMG_HEIGHT}x{config.IMG_WIDTH}x{config.CHANNELS}",
    }


def benchmark_inference(
    model_path: Path = config.DEFAULT_MODEL_PATH,
    iterations: int = 50,
    warmup_runs: int = 5,
) -> Dict:
    model_path = Path(model_path)
    if not model_path.exists():
        fallback = model_path.with_suffix(".h5")
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {fallback}")

    print(f"⚡ Benchmarking DefektGuard Inference Performance")
    print(f"   Model:        {model_path}")
    print(f"   Iterations:   {iterations}")
    print(f"   Warmup Runs:  {warmup_runs}")

    # Measure model load time
    t0 = time.perf_counter()
    model = tf.keras.models.load_model(model_path)
    load_time_ms = (time.perf_counter() - t0) * 1000
    print(f"   Cold Load Time: {load_time_ms:.2f} ms")

    # Generate dummy input tensor
    dummy_single = np.random.uniform(0.0, 1.0, size=(1, *config.INPUT_SHAPE)).astype(np.float32)

    # Warmup
    print("🔥 Warming up model...")
    for _ in range(warmup_runs):
        _ = model.predict(dummy_single, verbose=0)

    # Single-image latency benchmark
    print("⏱️ Running single-image latency benchmark...")
    latencies = []
    for _ in range(iterations):
        t_start = time.perf_counter()
        _ = model.predict(dummy_single, verbose=0)
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)

    latencies = np.array(latencies)
    single_results = {
        "iterations": iterations,
        "mean_ms": round(float(np.mean(latencies)), 2),
        "median_ms": round(float(np.median(latencies)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
        "throughput_fps": round(float(1000.0 / np.mean(latencies)), 2),
    }

    # Batch scaling benchmark
    batch_sizes = [1, 4, 8, 16, 32]
    batch_results = []
    print("📦 Running batch scaling benchmark...")

    for bs in batch_sizes:
        dummy_batch = np.random.uniform(0.0, 1.0, size=(bs, *config.INPUT_SHAPE)).astype(np.float32)
        # Warmup for batch
        _ = model.predict(dummy_batch, verbose=0)
        
        batch_times = []
        for _ in range(max(10, iterations // 3)):
            t_start = time.perf_counter()
            _ = model.predict(dummy_batch, verbose=0)
            t_end = time.perf_counter()
            batch_times.append((t_end - t_start) * 1000)
            
        mean_batch_ms = float(np.mean(batch_times))
        per_img_ms = mean_batch_ms / bs
        fps = (bs / (mean_batch_ms / 1000.0))

        batch_results.append({
            "batch_size": bs,
            "mean_batch_latency_ms": round(mean_batch_ms, 2),
            "per_image_latency_ms": round(per_img_ms, 2),
            "throughput_fps": round(fps, 2),
        })

    report = {
        "model_path": str(model_path),
        "model_parameters": int(model.count_params()),
        "model_load_time_ms": round(load_time_ms, 2),
        "single_image_benchmark": single_results,
        "batch_scaling_benchmark": batch_results,
        "environment": get_system_metadata(),
    }

    # Print summary
    print("\n" + "=" * 55)
    print("📊 INFERENCE BENCHMARK SUMMARY")
    print("=" * 55)
    print(f"Model Parameters:       {report['model_parameters']:,}")
    print(f"Cold Start Load Time:   {report['model_load_time_ms']} ms")
    print(f"Single-Image Mean Latency: {single_results['mean_ms']} ms")
    print(f"Single-Image Median:       {single_results['median_ms']} ms")
    print(f"Single-Image P95:          {single_results['p95_ms']} ms")
    print(f"Single-Image P99:          {single_results['p99_ms']} ms")
    print(f"Throughput (Batch 1):      {single_results['throughput_fps']} images/sec")
    print("-" * 55)
    print("Batch Scaling:")
    for b in batch_results:
        print(f"  Batch {b['batch_size']:2d}: {b['mean_batch_latency_ms']:6.2f} ms total | {b['per_image_latency_ms']:5.2f} ms/img | {b['throughput_fps']:6.2f} FPS")
    print("=" * 55 + "\n")

    save_path = config.REPORTS_DIR / "inference_benchmark.json"
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"💾 Saved inference benchmark to: {save_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DefektGuard Model Inference")
    parser.add_argument("--model-path", type=str, default=str(config.DEFAULT_MODEL_PATH), help="Path to trained model")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    args = parser.parse_args()

    benchmark_inference(model_path=Path(args.model_path), iterations=args.iterations, warmup_runs=args.warmup)

