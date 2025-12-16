#!/usr/bin/env python3
"""Grid search benchmark for mjwarp-render performance.

This script runs a grid search over different combinations of:
- nworld: number of parallel worlds (1, 2, 4, ..., 8192)
- resolution: width and height in pixels (16, 32, 64, ..., 512)

Results are saved to a JSON file for later visualization.

Usage:
    python benchmark_grid_search.py <model_path> [--output <output_file>] [--benchmark_frames <N>]
    python benchmark_grid_search.py --bowl_escape [--bowl_hsize <N>] [--bowl_vsize <F>] ...

Examples:
    # Benchmark with an MJCF model file
    python benchmark_grid_search.py /path/to/rodent.xml --output results.json

    # Benchmark with bowl escape environment (heightfield)
    python benchmark_grid_search.py --bowl_escape --output bowl_results.json

    # Bowl escape with custom parameters
    python benchmark_grid_search.py --bowl_escape --bowl_hsize 4 --bowl_vsize 3.0 --bowl_sigma 1.5
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_benchmark_output(output: str) -> dict | None:
    """Parse the benchmark output to extract metrics.

    Returns None if parsing fails (e.g., runtime error occurred).
    """
    try:
        results = {}

        # Extract images per second
        match = re.search(r"Images/second:\s+([\d.]+)", output)
        if match:
            results["images_per_second"] = float(match.group(1))

        # Extract throughput FPS
        match = re.search(r"Throughput:\s+([\d.]+)\s+FPS", output)
        if match:
            results["throughput_fps"] = float(match.group(1))

        # Extract average FPS
        match = re.search(r"Average:\s+([\d.]+)\s+FPS", output)
        if match:
            results["avg_fps"] = float(match.group(1))

        # Extract avg time per frame (ms)
        match = re.search(r"Avg per frame:\s+([\d.]+)\s+ms", output)
        if match:
            results["avg_time_ms"] = float(match.group(1))

        # Extract avg time per image (ms)
        match = re.search(r"Avg time/image:\s+([\d.]+)\s+ms", output)
        if match:
            results["avg_time_per_image_ms"] = float(match.group(1))

        # Extract total time
        match = re.search(r"Total time:\s+([\d.]+)\s+s", output)
        if match:
            results["total_time_s"] = float(match.group(1))

        if not results:
            return None

        return results
    except Exception as e:
        print(f"  Error parsing output: {e}")
        return None


def run_benchmark(
    nworld: int,
    resolution: int,
    benchmark_frames: int = 100,
    warmup_frames: int = 10,
    model_path: str | None = None,
    bowl_escape: bool = False,
    bowl_hsize: int = 2,
    bowl_vsize: float = 2.0,
    bowl_sigma: float = 1.25,
    bowl_amplitude: int = -10,
    bowl_seed: int = 0,
) -> dict:
    """Run a single benchmark with the given parameters.

    Returns a dict with the results or error information.
    """
    cmd = [
        "mjwarp-render",
    ]

    if bowl_escape:
        cmd.extend([
            "--bowl_escape",
            f"--bowl_hsize={bowl_hsize}",
            f"--bowl_vsize={bowl_vsize}",
            f"--bowl_sigma={bowl_sigma}",
            f"--bowl_amplitude={bowl_amplitude}",
            f"--bowl_seed={bowl_seed}",
        ])
    elif model_path is not None:
        cmd.append(model_path)

    cmd.extend([
        "--benchmark",
        f"--nworld={nworld}",
        f"--width={resolution}",
        f"--height={resolution}",
        f"--benchmark_frames={benchmark_frames}",
        f"--benchmark_warmup={warmup_frames}",
        "--depth=false",  # Only render RGB for speed
    ])

    result = {
        "nworld": nworld,
        "resolution": resolution,
        "benchmark_frames": benchmark_frames,
        "warmup_frames": warmup_frames,
        "success": False,
        "error": None,
    }

    try:
        print(f"  Running: nworld={nworld}, resolution={resolution}x{resolution}")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if proc.returncode != 0:
            # Check for common VRAM errors
            stderr = proc.stderr.lower()
            stdout = proc.stdout.lower()
            combined = stderr + stdout

            if "out of memory" in combined or "cuda" in combined or "vram" in combined:
                result["error"] = "VRAM_OOM"
                print(f"  VRAM out of memory")
            elif "runtime" in combined:
                result["error"] = "RuntimeError"
                print(f"  Runtime error")
            else:
                result["error"] = f"Exit code {proc.returncode}"
                print(f"  Failed with exit code {proc.returncode}")
            return result

        # Parse the output
        metrics = parse_benchmark_output(proc.stdout)
        if metrics:
            result["success"] = True
            result.update(metrics)
            print(f"  Success: {metrics.get('images_per_second', 'N/A'):.2f} images/sec")
        else:
            result["error"] = "ParseError"
            print(f"  Failed to parse output")

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout"
        print(f"  Timeout after 5 minutes")
    except Exception as e:
        result["error"] = str(e)
        print(f"  Exception: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Grid search benchmark for mjwarp-render performance"
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the MJCF XML model file (not required if --bowl_escape is used)",
    )
    parser.add_argument(
        "--bowl_escape",
        action="store_true",
        help="Use bowl escape environment with heightfield instead of MJCF file",
    )
    parser.add_argument(
        "--bowl_hsize",
        type=int,
        default=2,
        help="Horizontal size of the bowl (default: 2)",
    )
    parser.add_argument(
        "--bowl_vsize",
        type=float,
        default=2.0,
        help="Vertical size (depth) of the bowl (default: 2.0)",
    )
    parser.add_argument(
        "--bowl_sigma",
        type=float,
        default=1.25,
        help="Standard deviation of the Gaussian bump (default: 1.25)",
    )
    parser.add_argument(
        "--bowl_amplitude",
        type=int,
        default=-10,
        help="Amplitude of the Gaussian bump (default: -10)",
    )
    parser.add_argument(
        "--bowl_seed",
        type=int,
        default=0,
        help="Random seed for bowl heightfield generation (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--benchmark_frames",
        type=int,
        default=100,
        help="Number of frames to benchmark (default: 100)",
    )
    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=10,
        help="Number of warmup frames (default: 10)",
    )
    parser.add_argument(
        "--nworld_min",
        type=int,
        default=1,
        help="Minimum nworld value (default: 1)",
    )
    parser.add_argument(
        "--nworld_max",
        type=int,
        default=8192,
        help="Maximum nworld value (default: 8192)",
    )
    parser.add_argument(
        "--resolution_min",
        type=int,
        default=16,
        help="Minimum resolution (default: 16)",
    )
    parser.add_argument(
        "--resolution_max",
        type=int,
        default=512,
        help="Maximum resolution (default: 512)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.bowl_escape:
        model_path = None
        model_name = "bowl_escape"
    else:
        if args.model_path is None:
            print("Error: model_path is required unless --bowl_escape is used")
            sys.exit(1)
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)
        model_name = str(model_path)

    # Generate grid values (powers of 2)
    nworld_values = []
    val = args.nworld_min
    while val <= args.nworld_max:
        nworld_values.append(val)
        val *= 2

    resolution_values = []
    val = args.resolution_min
    while val <= args.resolution_max:
        resolution_values.append(val)
        val *= 2

    print("=" * 60)
    print("MJWARP-RENDER GRID SEARCH BENCHMARK")
    print("=" * 60)
    print(f"Model: {model_name}")
    if args.bowl_escape:
        print(f"  Bowl hsize: {args.bowl_hsize}")
        print(f"  Bowl vsize: {args.bowl_vsize}")
        print(f"  Bowl sigma: {args.bowl_sigma}")
        print(f"  Bowl amplitude: {args.bowl_amplitude}")
        print(f"  Bowl seed: {args.bowl_seed}")
    print(f"nworld values: {nworld_values}")
    print(f"Resolution values: {resolution_values}")
    print(f"Total combinations: {len(nworld_values) * len(resolution_values)}")
    print(f"Benchmark frames: {args.benchmark_frames}")
    print(f"Warmup frames: {args.warmup_frames}")
    print("=" * 60)

    # Store results
    metadata = {
        "model_path": str(model_path.absolute()) if model_path else None,
        "bowl_escape": args.bowl_escape,
        "timestamp": datetime.now().isoformat(),
        "nworld_values": nworld_values,
        "resolution_values": resolution_values,
        "benchmark_frames": args.benchmark_frames,
        "warmup_frames": args.warmup_frames,
    }
    if args.bowl_escape:
        metadata.update({
            "bowl_hsize": args.bowl_hsize,
            "bowl_vsize": args.bowl_vsize,
            "bowl_sigma": args.bowl_sigma,
            "bowl_amplitude": args.bowl_amplitude,
            "bowl_seed": args.bowl_seed,
        })

    results = {
        "metadata": metadata,
        "benchmarks": [],
    }

    # Track OOM to skip larger combinations
    oom_at_nworld = {}  # resolution -> nworld that first caused OOM

    total = len(nworld_values) * len(resolution_values)
    current = 0

    # Run grid search - iterate resolution first, then nworld
    # This way we can skip larger nworld values once OOM occurs for a resolution
    for resolution in resolution_values:
        print(f"\n--- Resolution: {resolution}x{resolution} ---")

        for nworld in nworld_values:
            current += 1
            print(f"\n[{current}/{total}]")

            # Skip if we've already hit OOM for smaller nworld at this resolution
            if resolution in oom_at_nworld and nworld >= oom_at_nworld[resolution]:
                print(f"  Skipping nworld={nworld} (OOM at nworld={oom_at_nworld[resolution]})")
                results["benchmarks"].append({
                    "nworld": nworld,
                    "resolution": resolution,
                    "success": False,
                    "error": "Skipped_OOM",
                })
                continue

            result = run_benchmark(
                nworld=nworld,
                resolution=resolution,
                benchmark_frames=args.benchmark_frames,
                warmup_frames=args.warmup_frames,
                model_path=str(model_path) if model_path else None,
                bowl_escape=args.bowl_escape,
                bowl_hsize=args.bowl_hsize,
                bowl_vsize=args.bowl_vsize,
                bowl_sigma=args.bowl_sigma,
                bowl_amplitude=args.bowl_amplitude,
                bowl_seed=args.bowl_seed,
            )
            results["benchmarks"].append(result)

            # Track OOM
            if result.get("error") == "VRAM_OOM":
                oom_at_nworld[resolution] = nworld

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path.absolute()}")

    # Print summary
    successful = sum(1 for b in results["benchmarks"] if b.get("success"))
    failed = len(results["benchmarks"]) - successful
    print(f"Successful benchmarks: {successful}")
    print(f"Failed benchmarks: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
