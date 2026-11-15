#This script encorporates the actual Ant Colony Optimisation nerve fibre tracking.
#ALL the settings that can be configured via the CLI are found here.

"""
BASICS OF HOW THE ANTS MOVE AND STOPPING CRITERIA:
The script works by allowing an ant to crawl around the most probabilistic areas where water is flowing within the brain.
It does this by calculating each voxels principle eigenvector, which probabilistically shows us the fiber direction of that voxel.
The ants can turn but they're limited to a cone around its current heading (view sphere.py).
A candidate solution is determined as impossible if the ant is outside the tracking mask, turns more than the limit, or lands where FA is below the threshold. This causes the ant to stop.

HOW ANTS SPAWN:
The ants are spawned from seeds which are placed in sections of high Fractional Anistrophy (where water moves a lot).

HOW AN ANT IS SCORED AND HOW GENERATIONS LEARN:
An ant is scored by calculating the sum of FA * cos(turn). This means a high scoring ant would go in the direction of flowing water, and turn smoothly.
Pheremones are deposited in high-scoring areas. Standard pheremone evaporation still applies.

TRACING:
At the end, the ants path is traced from every seed, both ways.

- By Hugo Piper, in sacrafice of his sleep schedule and parts of his social life.
"""

from __future__ import annotations
import copy
import math #Used for math written in C as its faster and quite frankly easier than reading 100 pages of NumPy documentation.
import sys
from dataclasses import dataclass
import numpy as np

from .ant_colony_optimisation import REPULSION100
from data_loading_and_math.measure_water_diffusion_at_voxel import local_tensor