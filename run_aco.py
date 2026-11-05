'''
3D MRI Diffuse Tensor Imaging tractography using Ant Colony optimisation.
Designed, written, tested & validated by I, Hugo Piper.

THIS IS THE PRIMARY SCRIPT FOR RUNNING THE ANT COLONY OPTIMISATION SOFTWARE.
There is no real logic processed here; only argument parsing, firing the shots to other scripts, and piping data.
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
        ("step", "step_size_voxel", "step length (voxels)"),
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
        ("tau-initial", "tau_initial", "pheromone where no ant has laid any"),
        ("tau-min", "tau_min", "pheromone never drops below value (maximum 1)"),
    ]),
]

def format_values(vals):
    if isinstance(vals, int):
        return f"{vals:,}"      #Formats integers, e.g: 1000000 turns into 1,000,000 
    else:
        return f"{vals:g}"      #Formats floats by removing trailing zeros. E.g: 0.1200000 turns into 0.12

#Basic code to build a command line interface. Turns the OPTIONS dictionary into genuine flags via argparse.
#I designed the arguments system this way to avoid a massive wall of repetitive code, doing it this way condenses everything into a for loop.
p = argparse.ArgumentParser(description="ACO fiber tracking on ACO-ready scan.")
p.add_argument("scan", help="The .nii.gz file which is ready to be processed")  #First mandatory positional argument which the path to the scan itself

for group, options in OPTIONS:
    g = p.add_argument(group)   #Adds each group name to argparse for display purposes (seeds & colony, how ants move...etc)
    for flag, name, text in options:
        print(f"flag = {flag}, name = {name}, text = {text}")
        continue