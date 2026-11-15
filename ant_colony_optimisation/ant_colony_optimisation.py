#This script encorporates the actual Ant Colony Optimisation nerve fibre tracking.
#ALL the settings that can be configured via the CLI are found here.

"""
BASICS OF HOW THE ANTS MOVE AND STOPPING CRITERIA:
The script works by allowing an ant to crawl around the most probabilistic areas where water is flowing within the brain.
It does this by calculating each voxels principle eigenvector, which probabilistically shows us the fiber direction of that voxel.
The ants can turn but they're limited to a cone around its current heading (view sphere.py) to 30deg.
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
from data_loading_and_math.measure_water_diffusion_at_voxel import (apparent_diffusivity, local_tensor, determine_processor, principal_direction_and_fa, tensor_times)

@dataclass
class ACOConfig:
    #DEFAULT SETTINGS OF CLI SETTINGS: Changing a lot of these dont really affect results a tonne, but I thought I'd include all of them for testing purposes.
    #Seeds and colony
    max_seeds: int = 10000           # [--streamlines] one streamline is traced from each seed: The more seeds the more streamlines (and computational cost)
    n_training_seeds: int = 15000    # [--training-seeds] seeds the exploring ants start from (at most max_seeds)
    seed_fa_threshold: float = 0.3   # [--seed-fa] seeds only where FA is at least this
    n_generations: int = 8           # [--generations] learning rounds (0 = no learning, also computationally expensive)
    n_ants_per_seed: int = 4         # [--ants] exploring ants per training seed, per generation

    #How the ants move
    step_size_vox: float = 0.5       # [--step] step length, in voxels
    max_turn_deg: float = 30.0       # [--max-turn] sharpest turn per step, in degrees
    n_candidate_dirs: int = 8        # [--candidates] directions tried in the turning cone (+ fibre and deflected)
    stop_fa_threshold: float = 0.2   # [--stop-fa] an ant stops where FA ahead is below this (not used with T1 maps)
    min_length_mm: float = 20.0      # [--min-length] shorter streamlines are dropped; ants under half this leave no pheromone
    max_length_mm: float = 250.0     # [--max-length] longest path (caps the number of steps)

    #Choice of weights:
    alpha: float = 1.0               # [--alpha] weight of the pheromone
    beta: float = 2.0                # [--beta] weight of the tensor (FA x diffusivity along the step)
    gamma: float = 10.0              # [--gamma] weight of going straight
    final_greediness: float = 3.0    # [--greediness] beta is multiplied by this for the final tracing

    #Pheromone and evaporation (Max-Min Ant System)
    evaporation_rate: float = 0.2    # [--evaporation] Share of the pheromone that evaporates each generation
    pheremone_initial: float = 0.1   # [--pheremone-initial] Pheromone where no ant has laid any
    pheremone_min: float = 0.02      # [--pheremone-min] pheromone never drops below this (the maximum is 1)

    #Not actual CLI variables, static variables for smoothing streamlines
    smooth_streamlines: bool = True  #Moving-average smoothing of the final streamlines, not intense.
    smoothing_window: int = 5        #Points averaged
    random_seed: int = 42            #Same seed + same settings = same result.

#CHUNKING because doing everything at once requires lots of RAM, which in 2026 is more expensive than gold.
TRAIN_CHUNK_SEEDS = 500        #seeds per work unit while the colony learns
EXTRACT_CHUNK_SEEDS = 1000     #seeds per work unit while tracing the streamlines
DIRECTION_LUT_RES = 256        #resolution of the direction: pheromone-bin table
STRONG_PHEREMONE = 0.5         #a 'strong' trail, mainly used for presentable statistics (might remove later??)

def make_cone_templates(n_dirs: int, max_angle_rad: float) -> np.ndarray:
    #Makes a cone around the ant so its limited in the direction it can go in and gives it basic options.
    #Based off standard mathematics.
    #The ant cant choose ANY angle from -30deg to 30deg because that would require coding a 3D engine which sounds like torture. view ant_tracking_direction_table.py
    #view direction_lookup_table function below for more info.
    out = [np.array([0.0, 0.0, 1.0])]
    n_extra = max(n_dirs - 1, 0)
    golden = np.pi * (3 - np.sqrt(5))
    for i in range(n_extra):
        theta = (i + 0.5) / n_extra * max_angle_rad
        phi = i * golden
        out.append(np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]))
    return np.array(out)

def rotate_cone(headings, templates):
    #Takes the above cone template and orientates it along the direction the ant is heading.
    #Stats: K,3 turned to point along each heading N,3. Returns N,K,3 unit vectors.
    xp = determine_processor(headings)
    e3 = headings
    zref = xp.asarray([0.0, 0.0, 1.0], dtype=e3.dtype)
    xref = xp.asarray([1.0, 0.0, 0.0], dtype=e3.dtype)
    ref = xp.where((xp.abs(e3[:, 2]) > 0.9)[:, None], xref, zref)
    e1 = xp.cross(ref, e3)
    e1 /= xp.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = xp.cross(e3, e1)
    cone = (templates[None, :, 0:1] * e1[:, None, :] +
            templates[None, :, 1:2] * e2[:, None, :] +
            templates[None, :, 2:3] * e3[:, None, :])
    cone /= xp.linalg.norm(cone, axis=2, keepdims=True) + 1e-12
    return cone

def direction_lookup_table(sphere: np.ndarray, res: int = DIRECTION_LUT_RES) -> np.ndarray:
    #Uses the 'ant_tracking_directions_table.py' script. We use this method partly because choosing any direction is hard in terms of 3D,
    #but mostly because its absurdley computationally expensive to try ALL the directions with very little accuracy gain.
    #Uses standardised mathematics.
    u = (np.arange(res) + 0.5) / res * 2 - 1
    gx, gy = np.meshgrid(u, u, indexing="ij")
    r2 = gx ** 2 + gy ** 2
    scale = np.where(r2 > 1, 1 / np.sqrt(np.maximum(r2, 1e-12)), 1.0)
    dx, dy = gx * scale, gy * scale
    dz = np.sqrt(np.clip(1 - dx ** 2 - dy ** 2, 0, 1))
    dirs = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
    return np.argmax(np.abs(dirs @ sphere.T), axis=1).reshape(res, res).astype(np.int32)

class World:
    #This is the ants 'world' that defines barriers, limits, directions change constraints...etc based on the variables above.
    #None of the variables here can actually be influenced or changed by the ants, like how we can't change the course of nature haha.

    def __init__(self, tensor_lt: np.ndarray, tracking_mask: np.ndarray, voxel_size, cfg: ACOConfig):
            mask = tracking_mask.astype(bool)
            #Scale the tensor so its median trace is 3 (FA and directions don't change), so the numbers don't depend on the scanner's units.
            trace = tensor_lt[..., 0] + tensor_lt[..., 2] + tensor_lt[..., 5]
            ref = float(np.median(trace[mask])) / 3.0 if mask.any() else 1.0
            if not np.isfinite(ref) or ref <= 0:
                ref = 1.0
            
            #All the scans data imported from 'load_and_read_scan_metadata.py'
            self.T = (np.nan_to_num(tensor_lt) / ref).astype(np.float32)
            self.mask = mask
            self.shape = mask.shape
            self.cfg = cfg
            self.voxel_size = np.asarray(voxel_size, dtype=np.float64)
            self.step_mm = cfg.step_size_vox * float(self.voxel_size.min())
            self.step_vox = self.step_mm / self.voxel_size          # (3,) voxels per step along each axis
            self.max_steps = int(math.ceil(cfg.max_length_mm / self.step_mm))   # no ant walks further than max length
            self.cos_max = math.cos(math.radians(cfg.max_turn_deg))              # turn limit, as a cosine
            self.templates = make_cone_templates(cfg.n_candidate_dirs, math.radians(cfg.max_turn_deg))   # the cone
            self.sphere = REPULSION100.copy()                       # the 100 pheromone direction bins
            self.n_bins = self.sphere.shape[0]
            self.bin_lut = direction_lookup_table(self.sphere)
            self.xp_name = "numpy"      # the array module the arrays live in (numpy or cupy)
            self.dtype = np.float64     # the ants' working precision
            self._set_index_helpers()
    
    @property
    def determine_processing_unit(self):
        #CPU or GPU (Nvidia only currently)
        return sys.modules[self.xp_name]

    def _set_index_helpers(self):
        #Checks if a position is a valid position inside the scan based on metadata
        dpu = self.determine_processing_unit
        self._hi = dpu.asarray(self.shape, dtype=dpu.intp) - 1
        self._upper = dpu.asarray(self.shape, dtype=self.dtype) - 0.5

    def to_device(self, xp, dtype=np.float32):
        #Copies over the entire 'world' class to the GPU to be processed if we have a valid GPU.
        wrld = copy.copy(self)
        wrld.T = xp.asarray(self.T, dtype=np.float32)
        wrld.mask = xp.asarray(self.mask)
        wrld.templates = xp.asarray(self.templates, dtype=dtype)
        wrld.step_vox = xp.asarray(self.step_vox, dtype=dtype)
        wrld.bin_lut = xp.asarray(self.bin_lut)
        wrld.xp_name = xp.__name__
        wrld.dtype = np.dtype(dtype).type
        wrld._set_index_helpers()
        return wrld

    def voxel_index(self, pos):
        #Translates voxel positions into integers
        dpu = self.determine_processing_unit
        return dpu.minimum(dpu.maximum(dpu.rint(pos).astype(dpu.intp), 0), self._hi)

    def inside(self, pos):
        #Returns true if a position is inside the brain (tracking) mask within the scan
        dpu = self.determine_processing_unit
        in_bounds = dpu.all((pos >= -0.5) & (pos < self._upper), axis=1)
        idx = self.voxel_index(pos)
        return in_bounds & self.mask[idx[:, 0], idx[:, 1], idx[:, 2]]

    def voxel_flat(self, pos):
        #Turns a voxel coordinate (x,y,z) into a single 1D number, part of the pheremone key system.
        idx = self.voxel_index(pos)
        _, Y, Z = self.shape
        return (idx[:, 0] * Y + idx[:, 1]) * Z + idx[:, 2]  #Basic mathematical formula to map out (x,y,z) in 1D space

    def direction_bins(self, dirs):
        #Finds which of the 100 directions an ants movement belongs to, so its pheremone can be looked up by future ants. (view ant_tracking_directions_table.py)
        dpu = self.determine_processing_unit
        flat = dirs.reshape(-1, 3)
        neg_or_pos = dpu.where(flat[:, 2] < 0, -1.0, 1.0).astype(flat.dtype)
        lookup_grid_size = DIRECTION_LUT_RES
        ix = dpu.clip(((flat[:, 0] * neg_or_pos + 1) * 0.5 * lookup_grid_size).astype(dpu.intp), 0, lookup_grid_size - 1)
        iy = dpu.clip(((flat[:, 1] * neg_or_pos + 1) * 0.5 * lookup_grid_size).astype(dpu.intp), 0, lookup_grid_size - 1)
        #Rounds the direction into a 'bin' (a bucket) based on the REUPULSION100 sphere.
        return self.bin_lut[ix, iy].reshape(dirs.shape[:-1])
    
