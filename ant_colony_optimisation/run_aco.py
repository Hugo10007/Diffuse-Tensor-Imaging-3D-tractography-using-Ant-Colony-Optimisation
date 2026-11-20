#This script is responsible for running the 'Ant Colony Optimisation' script in order.
#Also does light processing by dropping short or impossible streamlines, and lightly smooths the valid ones.
#I kept it seperate so the data acquisition (ACO) and calling the shots (this script) is kept seperate for readability purposes.

"""
THE RUNTIME ORDER IS AS FOLLOWS:
- IF T1 tissue maps in scan, have the ants walk only in white matter
- Run ACO pheremone learning if enabled, then trace one streamline for each seed.
- Reject impossibly short streamlines and streamlines outside the mask. Smooths the ones which are valid
- Keep streamlines whos end reach gray brain matter (Based on T1 scan)
TODO Save as a .trk file?????? Needs considering.
"""

from __future__ import  annotations

import os
import time
from dataclasses import dataclass, field
import numpy as np

from data_loading_and_math.t1_constrained_tracking import AnatomyConfig, active, constrain, end_filter
from multithreading_and_nvidia_gpu_support.processing_device import make_executor
from visualisation.streamline_tracing_and_filtering import apply_affine, filter_by_length, smooth, cluster_and_filter
from .ant_colony_optimisation import ACOConfig, World, run_aco, trail_stats

 
# ==========================================================================
# SETTINGS of a run (the colony's own settings are in colony.py)
# ==========================================================================

@dataclass
class RunConfig:
    aco: ACOConfig = field(default_factory=ACOConfig)               # the colony (colony.py)
    anatomy: AnatomyConfig = field(default_factory=AnatomyConfig)   # the T1 constraints (anatomy.py)
    output_dir: str = None        # folder for tractography.trk (None = don't save)
    device: str = "auto"          # "auto" (the GPU if usable), "gpu" or "cpu"
    n_workers: int = None         # CPU processes (None = CPU cores - 1)


def run_tracking(ready, cfg: RunConfig = None, log=None, progress=False):
    """Track an ACO-ready scan. ``log`` prints what happens (e.g. ``print``);
    ``progress`` shows progress bars. Returns a dict with the streamlines
    (world mm), their bundle labels, statistics and per-generation history."""
    cfg = cfg or RunConfig()
    log = log or (lambda *_: None)
    t_start = time.time()

    # --- where the ants may walk: white matter if the scan has T1 tissue maps
    full, aco = ready, cfg.aco
    use_t1 = active(full, cfg.anatomy)
    if use_t1:
        ready, aco = constrain(full, aco, cfg.anatomy)   # white-matter mask, FA floor instead of stop FA
    stats = {"input_file": full.path, "anatomical_constraints": use_t1,
             "tracking_mask_voxels": int(ready.tracking_mask.sum()),
             "seed_voxels": ready.seed_voxels(aco.seed_fa_threshold)}

    # --- the colony
    world = World(ready.tensor_lt, ready.tracking_mask, ready.voxel_size, aco)
    executor, stats["aco_device"] = make_executor(world, cfg.device, cfg.n_workers, log)
    history = []

    def on_generation(g, n, table, mean_path_mm):                # statistics after every generation
        reinforced, strong_pct = trail_stats(table, stats["tracking_mask_voxels"])
        history.append({"generation": g, "n_generations": n, "mean_path_mm": round(mean_path_mm, 2),
                        "pheromone_entries": len(table), "trail_voxels": reinforced, "strong_trail_pct": strong_pct})
        log(f"  generation {g}/{n}: mean ant path {mean_path_mm:5.1f} mm | strong trails {strong_pct:5.1f}% "
            f"of the mask | {len(table):,} pheromone entries | {time.time() - t0:5.0f} s")

    log(f"Colony ({stats['aco_device']})")
    t0 = time.time()
    try:
        raw = run_aco(world, ready.fa, executor, rng_seed=aco.random_seed, on_generation=on_generation, log=log,
                      progress=progress)
    finally:
        executor.close()
    stats["aco_seconds"] = round(time.time() - t0, 1)
    log(f"  traced {len(raw):,} streamlines (colony: {stats['aco_seconds']:.0f} s)")

    # --- clean-up
    log("Cleaning up")
    kept, lengths = filter_by_length(raw, ready.voxel_size, aco.min_length_mm, aco.max_length_mm)
    log(f"  length {aco.min_length_mm:g}-{aco.max_length_mm:g} mm: kept {len(kept):,} of {len(raw):,}")
    if aco.smooth_streamlines:
        kept = [smooth(s, aco.smoothing_window) for s in kept]
        log(f"  smoothed ({aco.smoothing_window}-point moving average)")
    if use_t1 and kept:
        idx = end_filter(kept, full, cfg.anatomy)
        stats["n_rejected_ends_not_in_grey_matter"] = len(kept) - len(idx)
        log(f"  T1 end filter: {len(idx):,} of {len(kept):,} reach grey matter (or the brain's edge) at both ends")
        kept, lengths = [kept[i] for i in idx], lengths[idx]
    streamlines = [apply_affine(s, ready.affine).astype(np.float32) for s in kept]     # voxels -> world mm

    labels = None
    if cfg.cleanup.enabled and streamlines:
        keep, labels, stats["n_bundles"] = cluster_and_filter(streamlines, cfg.cleanup, log=log)
        streamlines, lengths = [streamlines[i] for i in keep], lengths[keep]
    stats["n_streamlines_raw"] = len(raw)
    stats["n_streamlines_final"] = len(streamlines)
    if len(lengths):
        stats["length_mm_p10_p50_p90"] = [round(float(v), 1) for v in np.percentile(lengths, [10, 50, 90])]
