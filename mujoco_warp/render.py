# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""mjwarp-render: render an RGB and/or depth image from an MJCF.

Usage: mjwarp-render <mjcf XML path> [flags]

Examples:
  # Single frame rendering
  mjwarp-render benchmark/humanoid/humanoid.xml --nworld=1 --cam=0 --width=512 --height=512

  # Benchmark rendering speed (FPS)
  mjwarp-render benchmark/humanoid/humanoid.xml --benchmark --nworld=16 --benchmark_frames=100
"""

import sys
import time
from typing import Sequence

import mujoco
import numpy as np
import warp as wp
from absl import app
from absl import flags
from etils import epath
from PIL import Image

import mujoco_warp as mjw
from mujoco_warp._src.io import override_model
from mujoco_warp._src.render import render as render_frame

_NWORLD = flags.DEFINE_integer("nworld", 1, "number of parallel worlds")
_WORLD = flags.DEFINE_integer("world", 0, "world index to save from")
_CAM = flags.DEFINE_integer("cam", 0, "camera index to render")
_WIDTH = flags.DEFINE_integer("width", 512, "render width (pixels)")
_HEIGHT = flags.DEFINE_integer("height", 512, "render height (pixels)")
_RENDER_RGB = flags.DEFINE_bool("rgb", True, "render RGB image")
_RENDER_DEPTH = flags.DEFINE_bool("depth", True, "render depth image")
_USE_TEXTURES = flags.DEFINE_bool("textures", True, "use textures")
_USE_SHADOWS = flags.DEFINE_bool("shadows", False, "use shadows")
_DEVICE = flags.DEFINE_string("device", None, "override the default Warp device")
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "clear Warp kernel cache before rendering")
_OVERRIDE = flags.DEFINE_multi_string("override", [], "Model overrides (notation: foo.bar = baz)", short_name="o")
_OUTPUT_RGB = flags.DEFINE_string("output_rgb", "debug.png", "output path for RGB image")
_OUTPUT_DEPTH = flags.DEFINE_string("output_depth", "debug_depth.png", "output path for depth image")
_DEPTH_SCALE = flags.DEFINE_float("depth_scale", 5.0, "scale factor to map depth to 0..255 for preview")
_TILED = flags.DEFINE_bool("tiled", False, "render a 4x4 tiled grid across 16 worlds at 512x512")
_ROLLOUT = flags.DEFINE_bool("rollout", False, "render a rollout video instead of a single frame")
_NSTEPS = flags.DEFINE_integer("nstep", 128, "number of simulation steps in the rollout")
_ROLLOUT_OUTPUT = flags.DEFINE_string("output_video", "rollout.gif", "output path for rollout video")
_RANDOM_ACTIONS = flags.DEFINE_bool("random_actions", False, "apply random actions during rollout")

# Benchmark flags
_BENCHMARK = flags.DEFINE_bool("benchmark", False, "run rendering speed benchmark")
_BENCHMARK_FRAMES = flags.DEFINE_integer("benchmark_frames", 100, "number of frames to render for benchmark")
_BENCHMARK_WARMUP = flags.DEFINE_integer("benchmark_warmup", 10, "number of warmup frames before timing")

# Bowl escape environment flags
_BOWL_ESCAPE = flags.DEFINE_bool("bowl_escape", False, "use bowl escape environment with heightfield")
_BOWL_HSIZE = flags.DEFINE_integer("bowl_hsize", 2, "horizontal size of the bowl")
_BOWL_VSIZE = flags.DEFINE_float("bowl_vsize", 2, "vertical size (depth) of the bowl")
_BOWL_SIGMA = flags.DEFINE_float("bowl_sigma", 1.25, "standard deviation of the Gaussian bump")
_BOWL_AMPLITUDE = flags.DEFINE_integer("bowl_amplitude", -10, "amplitude of the Gaussian bump")
_BOWL_SEED = flags.DEFINE_integer("bowl_seed", 0, "random seed for bowl heightfield generation")


def _create_bowl_escape_model() -> mujoco.MjModel:
  """Create a bowl escape environment model with heightfield."""
  try:
    import jax
    from vnl_playground.tasks.rodent.bowl_escape import BowlEscape, default_config
  except ImportError as e:
    raise ImportError(
      "Bowl escape requires vnl-playground package. "
      "Install it from: /home/talmolab/Desktop/SalkResearch/vnl-playground"
    ) from e

  config = default_config()
  config.mujoco_impl = "warp"  # Required by base class compile()
  config.bowl_hsize = _BOWL_HSIZE.value
  config.bowl_vsize = _BOWL_VSIZE.value
  config.bowl_sigma = _BOWL_SIGMA.value
  config.bowl_amplitude = _BOWL_AMPLITUDE.value

  rng = jax.random.PRNGKey(_BOWL_SEED.value)
  print(f"Creating bowl escape environment...")
  print(f"  hsize={config.bowl_hsize}, vsize={config.bowl_vsize}")
  print(f"  sigma={config.bowl_sigma}, amplitude={config.bowl_amplitude}")
  print(f"  seed={_BOWL_SEED.value}")

  env = BowlEscape(rng=rng, config=config)
  return env.mj_model


def _load_model(path: epath.Path) -> mujoco.MjModel:
    if not path.exists():
        resource_path = epath.resource_path("mujoco_warp") / path
        if not resource_path.exists():
            raise FileNotFoundError(f"file not found: {path}\nalso tried: {resource_path}")
        path = resource_path

    print(f"Loading model from: {path}...")
    if path.suffix == ".mjb":
        return mujoco.MjModel.from_binary_path(path.as_posix())

    spec = mujoco.MjSpec.from_file(path.as_posix())
    # register SDF test plugins if present
    if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
        from mujoco_warp.test_data.collision_sdf.utils import register_sdf_plugins as register_sdf_plugins

        register_sdf_plugins(mjw)

    return spec.compile()


def _save_rgb_from_packed(packed_row: np.ndarray, width: int, height: int, out_path: str):
    packed = packed_row.reshape(height, width).astype(np.uint32)
    b = (packed & 0xFF).astype(np.uint8)
    g = ((packed >> 8) & 0xFF).astype(np.uint8)
    r = ((packed >> 16) & 0xFF).astype(np.uint8)
    img = Image.fromarray(np.dstack([r, g, b]))
    img.save(out_path)


def _save_depth(depth_row: np.ndarray, width: int, height: int, scale: float, out_path: str):
    arr = depth_row.reshape(height, width)
    arr = np.clip(arr / max(scale, 1e-6), 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    img.save(out_path)


def _rgb_image_from_packed(packed_row: np.ndarray, width: int, height: int) -> np.ndarray:
  """Convert a packed uint32 row into an (H, W, 3) uint8 RGB array."""
  packed = packed_row.reshape(height, width).astype(np.uint32)
  b = (packed & 0xFF).astype(np.uint8)
  g = ((packed >> 8) & 0xFF).astype(np.uint8)
  r = ((packed >> 16) & 0xFF).astype(np.uint8)
  return np.dstack([r, g, b])


def _depth_image_from_row(depth_row: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
  """Convert a depth row into an (H, W) uint8 array using the given scale."""
  arr = depth_row.reshape(height, width)
  arr = np.clip(arr / max(scale, 1e-6), 0.0, 1.0)
  return (arr * 255.0).astype(np.uint8)


def _save_tiled_rgb(
  packed_rows: np.ndarray,
  width: int,
  height: int,
  grid_rows: int,
  grid_cols: int,
  out_path: str,
):
  """Tile multiple RGB worlds into a single image and save it."""
  nworld = packed_rows.shape[0]
  expected = grid_rows * grid_cols
  if nworld < expected:
    raise ValueError(f"tiled rendering requires at least {expected} worlds, got {nworld}")

  tiles = []
  for wi in range(expected):
    tiles.append(_rgb_image_from_packed(packed_rows[wi], width, height))

  rows = []
  for r in range(grid_rows):
    row_tiles = tiles[r * grid_cols : (r + 1) * grid_cols]
    rows.append(np.concatenate(row_tiles, axis=1))
  full = np.concatenate(rows, axis=0)
  Image.fromarray(full).save(out_path)


def _save_tiled_depth(
  depth_rows: np.ndarray,
  width: int,
  height: int,
  scale: float,
  grid_rows: int,
  grid_cols: int,
  out_path: str,
):
  """Tile multiple depth worlds into a single image and save it."""
  nworld = depth_rows.shape[0]
  expected = grid_rows * grid_cols
  if nworld < expected:
    raise ValueError(f"tiled rendering requires at least {expected} worlds, got {nworld}")

  tiles = []
  for wi in range(expected):
    tiles.append(_depth_image_from_row(depth_rows[wi], width, height, scale))

  rows = []
  for r in range(grid_rows):
    row_tiles = tiles[r * grid_cols : (r + 1) * grid_cols]
    rows.append(np.concatenate(row_tiles, axis=1))
  full = np.concatenate(rows, axis=0)
  Image.fromarray(full).save(out_path)


def _sample_random_actions(m: mjw.Model, d: mjw.Data):
  """Sample random actions for all worlds and actuators."""
  nworld = d.nworld
  nu = m.nu

  if nu == 0:
    return

  # Get control ranges and limits
  ctrlrange = m.actuator_ctrlrange.numpy()  # shape: (*, nu, 2) or (nu, 2)
  ctrllimited = m.actuator_ctrllimited.numpy()  # shape: (nu,)

  # Handle heterogeneous vs homogeneous ctrlrange
  if ctrlrange.ndim == 3:
    # Heterogeneous: (nworld, nu, 2) - use world 0 as reference
    ctrl_lo = ctrlrange[0, :, 0]
    ctrl_hi = ctrlrange[0, :, 1]
  else:
    # Homogeneous: (nu, 2)
    ctrl_lo = ctrlrange[:, 0]
    ctrl_hi = ctrlrange[:, 1]

  # Sample random actions
  random_ctrl = np.random.uniform(size=(nworld, nu)).astype(np.float32)

  # Apply control ranges where limited
  for i in range(nu):
    if ctrllimited[i]:
      random_ctrl[:, i] = ctrl_lo[i] + random_ctrl[:, i] * (ctrl_hi[i] - ctrl_lo[i])
    else:
      # For unlimited actuators, use a reasonable default range [-1, 1]
      random_ctrl[:, i] = random_ctrl[:, i] * 2.0 - 1.0

  # Copy to device
  d.ctrl.assign(random_ctrl)


def _run_benchmark(
    m: "mjw.Model",
    d: "mjw.Data",
    rc: "mjw.RenderContext",
    num_frames: int,
    warmup_frames: int,
    nworld: int,
) -> dict:
  """Run rendering speed benchmark and return timing statistics.

  Args:
    m: MuJoCo Warp model.
    d: MuJoCo Warp data.
    rc: Render context.
    num_frames: Number of render calls (frames) to time.
    warmup_frames: Number of warmup frames before timing starts.
    nworld: Number of parallel worlds being rendered.

  Returns:
    Dictionary with benchmark results including FPS and timing stats.
  """
  # Warmup phase - render frames without timing to warm up GPU/kernels
  print(f"Running {warmup_frames} warmup frames...")
  for _ in range(warmup_frames):
    render_frame(m, d, rc)
  wp.synchronize()

  # Benchmark phase - time the rendering
  print(f"Benchmarking {num_frames} frames ({num_frames * nworld} total images across {nworld} worlds)...")
  frame_times = []

  for i in range(num_frames):
    start_time = time.perf_counter()
    render_frame(m, d, rc)
    wp.synchronize()  # Ensure GPU work is complete
    end_time = time.perf_counter()
    frame_times.append(end_time - start_time)

  # Calculate statistics
  frame_times = np.array(frame_times)
  total_time = np.sum(frame_times)
  avg_time = np.mean(frame_times)
  min_time = np.min(frame_times)
  max_time = np.max(frame_times)
  std_time = np.std(frame_times)

  # FPS for render calls (batched frames)
  avg_fps = 1.0 / avg_time if avg_time > 0 else 0
  min_fps = 1.0 / max_time if max_time > 0 else 0  # min FPS corresponds to max time
  max_fps = 1.0 / min_time if min_time > 0 else 0  # max FPS corresponds to min time
  throughput_fps = num_frames / total_time if total_time > 0 else 0

  # Total images accounting for all worlds
  total_images = num_frames * nworld
  images_per_second = total_images / total_time if total_time > 0 else 0
  avg_time_per_image = (avg_time * 1000) / nworld  # ms per individual image

  return {
    "num_frames": num_frames,
    "warmup_frames": warmup_frames,
    "nworld": nworld,
    "total_images": total_images,
    "total_time_s": total_time,
    "avg_time_ms": avg_time * 1000,
    "min_time_ms": min_time * 1000,
    "max_time_ms": max_time * 1000,
    "std_time_ms": std_time * 1000,
    "avg_time_per_image_ms": avg_time_per_image,
    "avg_fps": avg_fps,
    "min_fps": min_fps,
    "max_fps": max_fps,
    "throughput_fps": throughput_fps,
    "images_per_second": images_per_second,
  }


def _print_benchmark_results(results: dict, width: int, height: int):
  """Print benchmark results in a formatted way."""
  nworld = results["nworld"]
  print("\n" + "=" * 60)
  print("RENDERING BENCHMARK RESULTS")
  print("=" * 60)
  print(f"Configuration:")
  print(f"  Resolution:       {width} x {height}")
  print(f"  Parallel worlds:  {nworld}")
  print(f"  Warmup frames:    {results['warmup_frames']}")
  print(f"  Timed frames:     {results['num_frames']}")
  print(f"  Total images:     {results['total_images']} ({results['num_frames']} frames x {nworld} worlds)")
  print("-" * 60)
  print(f"Timing (per render call / batch of {nworld} images):")
  print(f"  Total time:       {results['total_time_s']:.3f} s")
  print(f"  Avg per frame:    {results['avg_time_ms']:.3f} ms")
  print(f"  Min per frame:    {results['min_time_ms']:.3f} ms")
  print(f"  Max per frame:    {results['max_time_ms']:.3f} ms")
  print(f"  Std deviation:    {results['std_time_ms']:.3f} ms")
  print("-" * 60)
  print(f"Performance (render calls / batched frames):")
  print(f"  Throughput:       {results['throughput_fps']:.2f} FPS")
  print(f"  Average:          {results['avg_fps']:.2f} FPS")
  print(f"  Min:              {results['min_fps']:.2f} FPS")
  print(f"  Max:              {results['max_fps']:.2f} FPS")
  print("-" * 60)
  print(f"Performance (individual images across all worlds):")
  print(f"  Images/second:    {results['images_per_second']:.2f}")
  print(f"  Avg time/image:   {results['avg_time_per_image_ms']:.3f} ms")
  print("=" * 60)


def _main(argv: Sequence[str]):
  # Handle bowl escape mode vs regular MJCF loading
  if _BOWL_ESCAPE.value:
    if len(argv) > 1:
      print("Warning: MJCF path ignored when --bowl_escape is enabled")
    mjm = _create_bowl_escape_model()
  else:
    if len(argv) < 2:
      raise app.UsageError("Missing required input: mjcf path (or use --bowl_escape).")
    elif len(argv) > 2:
      raise app.UsageError("Too many command-line arguments.")
    mjm = _load_model(epath.Path(argv[1]))

  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)

  wp.config.quiet = flags.FLAGS["verbosity"].value < 1
  wp.init()
  if _CLEAR_KERNEL_CACHE.value:
    wp.clear_kernel_cache()

  with wp.ScopedDevice(_DEVICE.value):
    m = mjw.put_model(mjm)

    if _OVERRIDE.value:
      override_model(m, _OVERRIDE.value)

    # Configure parallel worlds and per-camera resolution.
    if _TILED.value:
      # In tiled mode we always use 16 worlds and output a 4x4 grid at 512x512.
      nworld = 64
      grid_rows = 8
      grid_cols = 8
      final_width = 512 * 4
      final_height = 512 * 4
      render_width = final_width // grid_cols
      render_height = final_height // grid_rows
    else:
      nworld = int(_NWORLD.value)
      grid_rows = grid_cols = 1
      render_width = int(_WIDTH.value)
      render_height = int(_HEIGHT.value)

    d = mjw.put_data(mjm, mjd, nworld=nworld, njmax=700, nconmax=50)

    rc = mjw.create_render_context(
      mjm,
      m,
      d,
      (render_width, render_height),
      _RENDER_RGB.value,
      _RENDER_DEPTH.value,
      _USE_TEXTURES.value,
      _USE_SHADOWS.value,
      enabled_geom_groups=[0, 1, 2],
    )

    print(f"Model: ncam={m.ncam} nlight={m.nlight} ngeom={m.ngeom}\n")

    world = int(_WORLD.value)
    cam = int(_CAM.value)
    if cam < 0 or cam >= m.ncam:
      raise ValueError(f"camera index out of range: {cam} not in [0, {m.ncam - 1}]")
    if not _TILED.value:
      if world < 0 or world >= d.nworld:
        raise ValueError(f"world index out of range: {world} not in [0, {d.nworld - 1}]")

    cam_res = rc.cam_res.numpy()
    base_width = int(cam_res[cam][0])
    base_height = int(cam_res[cam][1])

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()

    if _BENCHMARK.value:
      # Benchmark mode - measure rendering performance
      results = _run_benchmark(
        m, d, rc,
        num_frames=_BENCHMARK_FRAMES.value,
        warmup_frames=_BENCHMARK_WARMUP.value,
        nworld=nworld,
      )
      _print_benchmark_results(results, render_width, render_height)
      return

    if _ROLLOUT.value:
      if not _RENDER_RGB.value:
        raise ValueError("rollout video requires RGB rendering to be enabled (--rgb).")

      # Use the physics timestep to choose how many simulation steps each
      # video frame should cover so that playback is approximately realtime.
      try:
        dt = float(m.opt.timestep.numpy()[0])
      except Exception:
        dt = 1.0 / 60.0

      target_fps = 30.0
      steps_per_frame = max(1, int(round(1.0 / (dt * target_fps))))
      frame_duration_ms = max(1, int(round(1000.0 / target_fps)))

      total_steps = int(_NSTEPS.value)
      action_info = " with random actions" if _RANDOM_ACTIONS.value else ""
      print(f"Rendering rollout for {total_steps} steps{action_info} (dt={dt:.4f}, steps_per_frame={steps_per_frame})...")
      frames = []

      step = 0
      while step < total_steps:
        render_frame(m, d, rc)

        if _TILED.value:
          rgb_all = rc.rgb_data.numpy()
          if rgb_adr[cam] != -1:
            slice_start = rgb_adr[cam]
            slice_end = slice_start + base_width * base_height
            rows = rgb_all[:, slice_start:slice_end]
            # Build a tiled frame from all worlds.
            expected = grid_rows * grid_cols
            if rows.shape[0] >= expected:
              tiles = []
              for wi in range(expected):
                tiles.append(_rgb_image_from_packed(rows[wi], base_width, base_height))
              row_imgs = []
              for r in range(grid_rows):
                row_tiles = tiles[r * grid_cols : (r + 1) * grid_cols]
                row_imgs.append(np.concatenate(row_tiles, axis=1))
              frame_array = np.concatenate(row_imgs, axis=0)
            else:
              frame_array = None
          else:
            frame_array = None
        else:
          rgb_all = rc.rgb_data.numpy()
          if rgb_adr[cam] != -1:
            slice_start = rgb_adr[cam]
            slice_end = slice_start + base_width * base_height
            row = rgb_all[world, slice_start:slice_end]
            frame_array = _rgb_image_from_packed(row, base_width, base_height)
          else:
            frame_array = None

        if frame_array is not None:
          frames.append(Image.fromarray(frame_array))

        # Advance simulation by the number of steps represented by this frame.
        for _ in range(steps_per_frame):
          if step >= total_steps:
            break
          if _RANDOM_ACTIONS.value:
            _sample_random_actions(m, d)
          mjw.step(m, d)
          step += 1

      if not frames:
        raise RuntimeError("no RGB frames were generated during rollout")

      frames[0].save(
        _ROLLOUT_OUTPUT.value,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
      )
      print(f"Saved rollout video to: {_ROLLOUT_OUTPUT.value}")
      return

    # Single-frame rendering path.
    print("Rendering single frame...")
    render_frame(m, d, rc)

    if _TILED.value:
      # Use all worlds and tile them into a 4x4 grid.
      if rgb_adr[cam] != -1:
        rgb_all = rc.rgb_data.numpy()
        slice_start = rgb_adr[cam]
        slice_end = slice_start + base_width * base_height
        rows = rgb_all[:, slice_start:slice_end]
        _save_tiled_rgb(rows, base_width, base_height, grid_rows, grid_cols, _OUTPUT_RGB.value)
        print(f"Saved tiled RGB to: {_OUTPUT_RGB.value}")

      if depth_adr[cam] != -1:
        depth_all = rc.depth_data.numpy()
        slice_start = depth_adr[cam]
        slice_end = slice_start + base_width * base_height
        rows = depth_all[:, slice_start:slice_end]
        _save_tiled_depth(
          rows,
          base_width,
          base_height,
          _DEPTH_SCALE.value,
          grid_rows,
          grid_cols,
          _OUTPUT_DEPTH.value,
        )
        print(f"Saved tiled depth to: {_OUTPUT_DEPTH.value}")
    else:
      # Original single-world behavior.
      if rgb_adr[cam] != -1:
        rgb = rc.rgb_data.numpy()
        row = rgb[world, rgb_adr[cam] : rgb_adr[cam] + base_width * base_height]
        _save_rgb_from_packed(row, base_width, base_height, _OUTPUT_RGB.value)
        print(f"Saved RGB to: {_OUTPUT_RGB.value}")

      if depth_adr[cam] != -1:
        depth = rc.depth_data.numpy()
        row = depth[world, depth_adr[cam] : depth_adr[cam] + base_width * base_height]
        _save_depth(row, base_width, base_height, _DEPTH_SCALE.value, _OUTPUT_DEPTH.value)
        print(f"Saved depth to: {_OUTPUT_DEPTH.value}")


def main():
    sys.argv[0] = "mujoco_warp.render"
    sys.modules["__main__"].__doc__ = __doc__
    app.run(_main)


if __name__ == "__main__":
    main()
