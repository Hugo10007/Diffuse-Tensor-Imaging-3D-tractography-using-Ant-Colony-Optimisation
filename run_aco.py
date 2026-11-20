
'''
3D MRI Diffuse Tensor Imaging tractography using Ant Colony optimisation.
Designed, written, tested & validated by I, Hugo Piper.

THIS IS THE PRIMARY SCRIPT FOR RUNNING THE ANT COLONY OPTIMISATION SOFTWARE.
There is no real logic processed here; only argument parsing (CLI frontend), firing the shots to other scripts, and piping data.
- View 'tracker.py' for a high level view of the ACO running.
- View 'aco_ready.py' for the actual ACO logic.
THIS IS A CLI BASED APPLICATION. USAGE BELOW.
'''

import argparse
import multiprocessing as mp
import os
import sys

#All the options that can be managed via the CLI as arguments.
# The layout is: printed_banner: argument_flag, matching_field(view_colony.py), help_text. 
OPTIONS = [
    ("Seeds and colony", [
        ("streamlines", "max_seeds", "one streamline is traced from each seed"),
        ("training-seeds", "n_training_seeds", "seeds the exploring ants start from"),
        ("seed-fa", "seed_fa_threshold", "seeds only where FA is at least value"),
        ("generations", "n_generations", "learning rounds (0 = no learning)"),
        ("ants", "n_ants_per_seed", "ants that explore per training seed, per generation"),
    ]),
    ("How the ants move", [
        ("step", "step_size_vox", "step length (voxels)"),
        ("max-turn", "max_turn_deg", "sharpest turn per step (degrees)"),
        ("candidates", "n_candidate_directions", "directions tried in the turning cone"),
        ("stop-fa", "stop_fa_threshold", "an ant stops where FA ahead is below value"),
        ("min-length", "min_length_mm", "minimum allowed path (mm)"),
        ("max-length", "max_length_mm", "longest allowed path (mm)"),
    ]),
    ("Choice weights: alpha, beta, gamma", [
        ("alpha", "alpha", "weight of the pheromone"),
        ("beta", "beta", "weight of the tensor (FA * diffusivity along the step)"),
        ("gamma", "gamma", "weight of going straight"),
        ("greediness", "final_greediness", "beta is multiplied by value for the final tracing"),
    ]),
    ("Pheromone (Max-Min Ant System)", [
        ("evaporation", "evaporation_rate", "rho: share that evaporates each generation"),
        ("pheromone-initial", "pheromone_initial", "pheromone where no ant has laid any"),
        ("pheromone-min", "pheromone_min", "pheromone never drops below value (maximum 1)"),
    ]),
]
LINE = "=" * 92

def format_values(vals):
    if isinstance(vals, int):
        return f"{vals:,}"      #Formats integers, e.g: 1000000 turns into 1,000,000 
    else:
        return f"{vals:g}"      #Formats floats by removing trailing zeros. E.g: 0.1200000 turns into 0.12

#Basic code to build a command line interface. Turns the OPTIONS dictionary into genuine flags via argparse.
#I designed the arguments system this way to avoid a massive wall of repetitive code, doing it this way condenses everything into a for loop.

def main():
    from data_loading_and_math import t1_constrained_tracking
    from data_loading_and_math import load_and_read_scan_metadata
    from data_loading_and_math.t1_constrained_tracking import AnatomyConfig
    from ant_colony_optimisation.ant_colony_optimisation import ACOConfig
    from multithreading_and_nvidia_gpu_support.processing_device import default_workers, gpu_info
    from visualisation.streamline_tracing_and_filtering import CleanupConfig
    from ant_colony_optimisation.run_aco import RunConfig, run_tracking

    # --- the command line: one flag per colony setting, defaults from ACOConfig
    defaults = ACOConfig()
    p = argparse.ArgumentParser(description="Ant colony fibre tracking on an ACO-ready scan (made by DTI Prep).")
    p.add_argument("scan", help="the *_aco_ready.nii.gz file")
    for group, options in OPTIONS:
        g = p.add_argument_group(group)
        for flag, name, text in options:
            default = getattr(defaults, name)
            g.add_argument("--" + flag, dest=name, type=type(default), default=default,
                           metavar="N" if isinstance(default, int) else "X", help=f"{text} (default {format_values(default)})")
    g = p.add_argument_group("Run")
    g.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                   help="auto = the NVIDIA GPU if usable, else the CPU (default auto)")
    g.add_argument("--workers", type=int, default=default_workers(), metavar="N",
                   help=f"CPU processes when running on the CPU (default {default_workers()})")
    g.add_argument("--no-viewer", action="store_true", help="don't open the 3D window at the end")
    args = p.parse_args()

    aco = ACOConfig(**{name: getattr(args, name) for _, options in OPTIONS for _, name, _ in options})

    # --- the scan
    try:
        ready = load_and_read_scan_metadata.load(args.scan)
    except (OSError, ValueError) as exc:
        sys.exit(f"Can't open {args.scan}: {exc}")
    t1, cleanup = AnatomyConfig(), CleanupConfig()
    use_t1 = t1_constrained_tracking.active(ready, t1)
    walk = t1_constrained_tracking.white_matter_mask(ready, t1) if use_t1 else ready.tracking_mask    # where the ants may walk
    n_seed_voxels = int((walk & (ready.fa >= aco.seed_fa_threshold)).sum())

    # --- where it runs
    device, where = "cpu", f"CPU, {args.workers} process{'es' if args.workers > 1 else ''}"
    if args.device != "cpu":
        ok, desc = gpu_info()
        if ok:
            device, where = "gpu", f"GPU, {desc}"
        else:
            where += f" (no GPU: {desc})"

    # --- what is about to run
    X, Y, Z = ready.shape
    vs = ready.voxel_size
    print(LINE)
    print(f"Scan      {os.path.basename(args.scan)}")
    print(f"          {X} x {Y} x {Z} voxels of {vs[0]:.2f} x {vs[1]:.2f} x {vs[2]:.2f} mm")
    if use_t1:
        print(f"T1 maps   yes: the ants walk only in white matter ({int(walk.sum()):,} voxels), stop only where "
              f"FA < {t1.fa_floor:g},")
        print("          and both ends of a streamline must reach grey matter")
    else:
        print(f"T1 maps   no: the ants walk in the tracking mask ({int(walk.sum()):,} voxels) and stop where "
              f"FA < {aco.stop_fa_threshold:g}")
    print(f"Seeds     {aco.max_seeds:,} streamlines from {n_seed_voxels:,} voxels with FA >= {aco.seed_fa_threshold:g}")
    print(f"Runs on   {where}")
    print()
    print("Settings (change one with its flag, e.g. --generations 12, or the defaults in acotrack/colony.py)")
    for group, options in OPTIONS:
        print(f"  {group}")
        for flag, name, text in options:
            note = ""
            if name == "n_training_seeds" and aco.n_training_seeds > aco.max_seeds:
                note = f"  [{aco.max_seeds:,} used: at most --streamlines]"
            if name == "stop_fa_threshold" and use_t1:
                note = f"  [not used: T1 maps, FA floor {t1.fa_floor:g}]"
            print(f"    --{flag:<16}{format_values(getattr(aco, name)):>9}   {text}{note}")
    print(f"  Fixed (code only): smoothing {aco.smoothing_window} points | bundle filter "
          f"{cleanup.cluster_threshold_mm:g} mm, bundles of >= {cleanup.min_cluster_size} | random seed {aco.random_seed}"
          + (f" | T1 white matter >= {t1.wm_threshold:g}" if use_t1 else ""))
    print(LINE)

    # --- run
    from tqdm import tqdm
    cfg = RunConfig(aco=aco, cleanup=cleanup, anatomy=t1, device=device, n_workers=args.workers)
    try:
        res = run_tracking(ready, cfg, log=tqdm.write, progress=True)
    except KeyboardInterrupt:
        sys.exit("\nStopped.")
    s = res["stats"]

    print(LINE)
    bundles = f" in {s['n_bundles']:,} bundles" if "n_bundles" in s else ""
    print(f"Result    {s['n_streamlines_final']:,} streamlines{bundles}")
    if "length_mm_p10_p50_p90" in s:
        p10, p50, p90 = s["length_mm_p10_p50_p90"]
        print(f"Lengths   {p10:.0f} / {p50:.0f} / {p90:.0f} mm (10th / 50th / 90th percentile)")
    print(f"Time      colony {s['aco_seconds']:.0f} s, total {s['total_seconds']:.0f} s")
    print(LINE)

    if not args.no_viewer and res["streamlines_world"]:
        print("Opening the 3D view (close the window to finish)...")
        from visualisation.live_viewer import show
        show(res["streamlines_world"], title=f"ACO Tracker: {os.path.basename(args.scan)}")


if __name__ == "__main__":
    mp.freeze_support() #Worker processes on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    main()
