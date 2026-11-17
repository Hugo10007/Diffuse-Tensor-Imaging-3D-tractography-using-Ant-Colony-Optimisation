#This script encorporates the actual Ant Colony Optimisation nerve fibre tracking.
#ALL the settings that can be configured via the CLI are found here.
#All pheremone logic, ACO logic, and data chunking found here (leave data chunking active unless you have a Sam Altman amount of RAM)

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
    

class PheromoneTable:
    #This class is responsible for the pheromone level (tau) for every MRI voxel and direction bin that has been reinforced.
    #Data is stored as sorted keys (voxel * n_bins + bin) and values. 
    #Anything that has not been stored has a ''tau_initial'' default starting value which is determined by the 'tau_initial' parameter.

    def __init__(self, n_bins, tau_initial, tau_min, keys=None, vals=None, query_dtype=np.float64):
        self.n_bins = n_bins
        self.tau_initial = float(tau_initial)
        self.tau_min = float(tau_min)
        self.keys = np.zeros(0, np.int64) if keys is None else keys
        self.vals = np.zeros(0, np.float32) if vals is None else vals
        self.query_dtype = query_dtype

    def __len__(self):
        return len(self.keys)

    def to_device(self, xp, query_dtype=np.float32):
        #Creates a read-only copy so the GPU can process ants incase the user chooses to. Only works on Nvidia GPU's as of now due to time constraint
        return PheromoneTable(self.n_bins, self.tau_initial, self.tau_min, xp.asarray(self.keys), xp.asarray(self.vals), query_dtype)

    def query(self, keys):
        #Determines the pheremone level for each voxel based on a key as input, used to quickly look them up to avoid repetitive sloppy code everywhere
        processor = determine_processor(keys)
        if len(self.keys) == 0:
            return processor.full(keys.shape, self.tau_initial, dtype=self.query_dtype)
        flat = keys.ravel()
        i = processor.clip(processor.searchsorted(self.keys, flat), 0, len(self.keys) - 1)
        found = self.keys[i] == flat
        out = processor.where(found, self.vals[i], self.tau_initial).astype(self.query_dtype)
        return out.reshape(keys.shape)

    def update(self, dep_keys, dep_amounts, rho):
        #This applies a single entire generations pheremone update once the ants have finished their exploring and laid pheremones.
        #The equation used to update the pheremone table is:
        # τ_new(v, b) = clip( (1 − ρ) · τ_old(v, b) + ρ · Δτ(v, b), τ_min, 1 )
        #Where τ_old is the old pheremone, (v, b) is the voxel and direction bin, ρ is the evaporation rate, Δτ for voxels with no update, and clip limits limits the growth and evaporation to reasonable levels incase of limitation failure.
        if len(dep_keys):
            voxel = dep_keys // self.n_bins                                  #The voxel of each key
            starts = np.flatnonzero(np.r_[True, voxel[1:] != voxel[:-1]])     #Where each voxel's keys start
            vmax = np.maximum.reduceat(dep_amounts, starts)                   #The largest deposit per voxel
            counts = np.diff(np.r_[starts, len(dep_keys)])
            delta = dep_amounts / np.repeat(np.maximum(vmax, 1e-30), counts)  #Scaled to [0, 1] per voxel.
        else:
            delta = dep_amounts
        all_keys = np.union1d(self.keys, dep_keys) #Gets the old trails + new deposits
        old = self.query(all_keys) #Tau_initial for new trails found
        d = np.zeros(len(all_keys)) #delta = 0 where nothing was deposited (Δτ)
        if len(dep_keys):
            d[np.searchsorted(all_keys, dep_keys)] = delta
        new = (1.0 - rho) * old + rho * d #Evaporate, then reinforce if pheremone level is high.
        self.keys = all_keys
        self.vals = np.clip(new, self.tau_min, 1.0).astype(np.float32) #Clipping to Max-Min bounds.



def candidate_directions(world: World, pos, heading):
    #Returns the possible candidate directions an ant can choose FOR EVERY STEP.
    #Takes the world, position of the ant, and heading of the ant to choose candiate directions within a 30deg cone.
    #Based on processed MRI metadata of tensor FA values.
    processor = world.xp
    comps, _, principal, _ = local_tensor(world.T, pos)                  # the tensor where the ant stands
    forwards = processor.sign(processor.einsum("ij,ij->i", principal, heading))      # flip the fibre direction to point ahead
    forwards = processor.where(forwards == 0, 1.0, forwards).astype(world.dtype)
    deflected = tensor_times(comps, heading)                            # D @ heading
    deflected /= processor.linalg.norm(deflected, axis=1, keepdims=True) + 1e-12
    cand = processor.empty((len(pos), len(world.templates) + 2, 3), dtype=world.dtype)
    cand[:, 0] = principal * forwards[:, None]
    cand[:, 1] = deflected
    cand[:, 2:] = rotate_cone(heading, world.templates)
    return cand


def score_candidate_moves(world: World, table: PheromoneTable, pos, heading, cand, target, beta):
    #Scores every candidate move as pheromone^alpha × heuristic^beta × straightness^y.
    #sets a moves score to 0 if it turns too sharply, lands in low FA or leaves the brain mask.

    cfg, processor = world.cfg, world.xp
    n, k = cand.shape[:2]
    target_flat = target.reshape(-1, 3)
    comps, evals, _, fa = local_tensor(world.T, target_flat) #The tensor where each move lands
    
    #How freely water diffuses along the move, relative to the easiest
    adc_rel = apparent_diffusivity(comps, cand.reshape(-1, 3)) / processor.maximum(evals[:, 0], 1e-12)
    fa = fa.reshape(n, k)
    cos_turn = processor.einsum("nkj,nj->nk", cand, heading) #Cosine of each move's turn to enforce the turn limit.

    #Checks if an ant did a valid move, scores 0 otherwise
    possible = ((cos_turn >= world.cos_max - 1e-9) & (fa >= cfg.stop_fa_threshold) &    #Turn limit and FA stop
                world.inside(target_flat).reshape(n, k))                                #Ensures ant is in the brain mask
    eta = fa * processor.clip(adc_rel, 0.0, 1.0).reshape(n, k)                 #Heuristic calculation: FA x relative diffusivity
    keys = world.voxel_flat(pos)[:, None].astype(processor.int64) * world.n_bins + world.direction_bins(cand)
    tau = table.query(keys) #Queries pheremone lookup on voxel and direction
    w = (tau ** cfg.alpha) * (processor.maximum(eta, 1e-12) ** beta) * (processor.clip(cos_turn, 1e-6, 1.0) ** cfg.gamma)
    return processor.where(possible, w, 0.0), fa, cos_turn, keys               #Scores impossible moves a weight of 0


def roulette(weights, total, rng):
    #FOR ENCOURAGING EXPLORING ANTS TO PICK ITS NEXT MOVE AT RANDOM. Without this the ants kept taking the same path and giving me a headache.
    #Picks one move per ant at random, with high scoring moves having a higher chance of being chosen.
    #Same logic as a roulette wheel, except each slice differs in size based on score. 
    #This logic is only used during learning when greedy = False.
    processor = determine_processor(weights)
    cumulativesum = processor.cumsum(weights, axis=1) #Running total along each row of weights
    r = rng.random(len(weights)) * total #Chooses a random weight from 0 to n 
    return processor.minimum((cumulativesum < r[:, None]).sum(axis=1), weights.shape[1] - 1) #Returns the move it falls on



#PROCESSES THE FIRST BATCHES OF ANTS
def run_ants(world: World, table: PheromoneTable, start_pos, start_dir, rng=None,
             greedy=False, record_keys=False, record_paths=False):
    #Walks a batch of ants until none can move.
    #greedy: Always takes the best move (used for final tracing, disabled roulette logic)
    #quality: sum of fa * cos(turn). Calculates the quality of a path taken.
    #record_paths: Records the points (voxels) visited
    cfg, processor, dt = world.cfg, world.xp, world.dtype
    n = len(start_pos)
    position = processor.array(start_pos, dtype=dt)
    heading = processor.array(start_dir, dtype=dt)
    heading /= processor.linalg.norm(heading, axis=1, keepdims=True) + 1e-12
    alive = processor.ones(n, dtype=bool) #Still walking?
    n_steps = processor.zeros(n, dtype=processor.int32) #Steps taken for an ant
    quality = processor.zeros(n, dtype=dt) #path score of an ant: sum of FA * cos(turn)
    beta = cfg.beta * (cfg.final_greediness if greedy else 1.0) #Disables roulette for the final tracing
    rec_owner, rec_key, rec_pos = [], [], []

    for _ in range(world.max_steps):
        act = processor.flatnonzero(alive) #Are ants still walking??
        if act.size == 0:
            break
        p, h = position[act], heading[act]
        cand = candidate_directions(world, p, h) #Possible directions an ant can choose
        target = p[:, None, :] + cand * world.step_vox[None, None, :] #where each ant lands
        w, fa, cos_turn, keys = score_candidate_moves(world, table, p, h, cand, target, beta) #Scores the candidate moves
        total = w.sum(axis=1)
        go = total > 0 #Ants with at least one possible move
        choice = processor.argmax(w, axis=1) if greedy else roulette(w, total, rng)#Best move, or a random one ~ p IF greediness is false. Uses roulette logic.

        rows = processor.arange(len(act))
        moved = act[go]
        new_pos = target[rows, choice][go]
        position[moved] = new_pos #makes the ant actually take the step
        heading[moved] = cand[rows, choice][go]
        n_steps[moved] += 1
        quality[moved] += (fa[rows, choice] * cos_turn[rows, choice])[go] #Score the step
        alive[act[~go]] = False #if no possible move, stop the ant
        if record_keys: #Records pheromone deposit
            rec_owner.append(moved)
            rec_key.append(keys[rows, choice][go])
        if record_paths: #Records the path for the streamlines
            if not record_keys:
                rec_owner.append(moved)
            rec_pos.append(new_pos.astype(processor.float32))

    #Displays nerd data for the user
    out = {"n_steps": n_steps, "quality": quality}
    owner = processor.concatenate(rec_owner) if rec_owner else processor.zeros(0, processor.intp)
    if record_keys:
        out["keys"] = processor.concatenate(rec_key) if rec_key else processor.zeros(0, processor.int64)
        out["owner"] = owner
    if record_paths:
        pts = processor.concatenate(rec_pos) if rec_pos else processor.zeros((0, 3), processor.float32)
        #Groups by ant, keeping step order (records were appended step by step)
        if processor is np:
            order = np.argsort(owner, kind="stable")
        else:   #If using GPU, unique composite key: any sort gives the stable order
            order = processor.argsort(owner.astype(processor.int64) * (len(owner) + 1) + processor.arange(len(owner)))
        out["points"] = pts[order]
        out["owner_pts"] = owner[order]
    return out

#SETTING UP EXPLORING ANTS
#RUNS ON WORK UNITS: one chunk runs here, in worker processes, or batched on a GPU.
def exploring_ants(world: World, seeds: np.ndarray, rng):
    #Sets up the exploring ants, uses a random seed.
    m = world.cfg.n_ants_per_seed
    principal, _ = principal_direction_and_fa(world.T, seeds)
    starts = np.repeat(seeds, m, axis=0)
    dirs = np.repeat(principal, m, axis=0)
    signs = np.tile(np.where(np.arange(m) % 2 == 0, 1.0, -1.0), len(seeds))
    dirs = signs[:, None] * dirs + rng.normal(scale=0.1, size=dirs.shape)
    return starts, dirs


def train_chunk(world: World, table: PheromoneTable, seeds: np.ndarray, rng_seed: int):
    #Runs one generation of exploring ants based on a random seed.
    #Returns their pheromone deposit as a key, and mean path length in the amount of steps the ant took.
    rng = np.random.default_rng(rng_seed)
    starts, dirs = exploring_ants(world, seeds, rng)
    res = run_ants(world, table, starts, dirs, rng=rng, record_keys=True)
    score = res["quality"].copy() #Each ant's path score
    score[res["n_steps"] * world.step_mm < 0.5 * world.cfg.min_length_mm] = 0.0 #if path is too short, no pheromone applied
    amounts = score[res["owner"]] #Every step deposits its ant's score
    keep = amounts > 0
    mean_steps = float(res["n_steps"].mean())
    if not np.any(keep):
        return np.zeros(0, np.int64), np.zeros(0), mean_steps
    keys, inverse = np.unique(res["keys"][keep], return_inverse=True)
    return keys, np.bincount(inverse, weights=amounts[keep]), mean_steps #The total deposit per voxel and direction


def join_both_ways(seeds: np.ndarray, forward_paths, backward_paths):
    #Joins a nerve path both ways. This is because a seed may not start at the very beginning of a nerve fiber
    #meaning we need to connect the single nerve fiber that the ants found in both + and - directions. Its wacky if you dont understand tractography but makes sense to me
    seeds32 = seeds.astype(np.float32)
    parts = [np.concatenate([back[::-1], seeds32[i:i + 1], fwd], axis=0)
             for i, (fwd, back) in enumerate(zip(forward_paths, backward_paths))]
    return np.concatenate(parts, axis=0), np.array([len(p) for p in parts], dtype=np.int64)


def split_paths(res):
    #Converts a large list of recorded points for all ants into 1 'path' per ant
    return np.split(res["points"], np.cumsum(res["n_steps"])[:-1])


def extract_chunk(world: World, table: PheromoneTable, seeds: np.ndarray, rng_seed: int = 0):
    #Traces one streamline per seed, both ways. view join_both_ways() for more info.
    #Always takes the best path the ants found, no roulette loogic here.

    principal, _ = principal_direction_and_fa(world.T, seeds)
    forward = run_ants(world, table, seeds, principal, greedy=True, record_paths=True)     # one way
    backward = run_ants(world, table, seeds, -principal, greedy=True, record_paths=True)   # the other way
    return join_both_ways(seeds, split_paths(forward), split_paths(backward))


#SETTING UP THE COLONY
def generate_seeds(fa: np.ndarray, tracking_mask: np.ndarray, cfg: ACOConfig, rng: np.random.Generator):
    #Generates seeds at random points within the tracking (brain) mask. Seeds MUST be placed on voxels with Fractional Anistrophy of at least the seed threshold.
    #In cases where there are more seeds than voxels, a voxel may get multiple seeds at different spots using trilinear interpolation logic.
    cand = np.argwhere(tracking_mask & (fa >= cfg.seed_fa_threshold))
    if len(cand) == 0:
        return np.zeros((0, 3))
    n = cfg.max_seeds
    idx = rng.choice(len(cand), size=n, replace=n > len(cand))
    return cand[idx].astype(np.float64) + rng.uniform(-0.49, 0.49, size=(n, 3))


def merge_deposits(results):
    #Adds up the pheremone deposits of all the ant batches after completion.
    #Each voxel and direction key ends up with a single total value
    parts = [r for r in results if len(r[0])]
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0)
    keys, inverse = np.unique(np.concatenate([r[0] for r in parts]), return_inverse=True)
    return keys, np.bincount(inverse, weights=np.concatenate([r[1] for r in parts]))


def trail_stats(table: PheromoneTable, n_mask_voxels: int):
    #Reports which trails have strong pheremones/reinforcement based on the 'STRONG_PHEREMONE' value. 
    voxel = table.keys // table.n_bins
    reinforced = len(np.unique(voxel[table.vals > table.tau_initial]))
    strong = len(np.unique(voxel[table.vals >= STRONG_PHEREMONE]))
    return reinforced, round(100.0 * strong / max(n_mask_voxels, 1), 2)


def _chunks(seeds, size):
    #Splits seeds into equal batches so ants can run in parallel on the CPU
    return [seeds[i:i + size] for i in range(0, len(seeds), size)]

#RUNS THE FULL ACO ALGORITHM
def run_aco(world: World, fa: np.ndarray, executor, rng_seed: int = 42, on_generation=None, log=None,
            progress=False):
    #Runs everything, lets the colony learn, then traces the colonies finding using streamlines.
    #Uses work units to run in parallel using train_chunk and extract_chunk, or on the GPU if available.
    #Prints logs and messages to the user so they dont get bored. Also shows a fancy progress bar 

    from tqdm import tqdm #Fancy loading bar module which reminds me of the early 2000's.
    
    log = log or (lambda *_: None)
    cfg = world.cfg
    seeds = generate_seeds(fa, world.mask, cfg, np.random.default_rng(rng_seed))
    if len(seeds) == 0:
        log("No seed voxels: nothing above the seed FA threshold inside the tracking mask.")
        return []
    n_train = min(len(seeds), cfg.n_training_seeds) if cfg.n_training_seeds else len(seeds)
    log(f"  learning: {cfg.n_generations} generations x {n_train:,} training seeds x {cfg.n_ants_per_seed} ants")

    #1) LEARNING OF THE COLONY: every generation the exploring ants lay pheromones and the pheremone table is updated
    table = PheromoneTable(world.n_bins, cfg.tau_initial, cfg.tau_min)
    train_seeds = _chunks(seeds[:n_train], TRAIN_CHUNK_SEEDS)
    bar = tqdm(total=cfg.n_generations * len(train_seeds), desc="Learning", unit="chunk", disable=not progress,
               leave=False)
    for g in range(cfg.n_generations):
        jobs = [(s, rng_seed * 1_000_003 + g * 10_007 + i) for i, s in enumerate(train_seeds)]   # fixed random streams
        results = executor.run("train", table, jobs, lambda i, r: bar.update(1))
        table.update(*merge_deposits(results), cfg.evaporation_rate)
        mean_path_mm = np.mean([r[2] for r in results]) * world.step_mm
        if on_generation is not None:
            on_generation(g + 1, cfg.n_generations, table, float(mean_path_mm))
    bar.close()

    #2) TRACING THE COLONIES RESULTS: one greedy streamline per seed, both ways. No roulette due to greediness being a necessity.
    log(f"  tracing: one streamline from each of the {len(seeds):,} seeds, both ways")
    jobs = [(s, 0) for s in _chunks(seeds, EXTRACT_CHUNK_SEEDS)]
    bar = tqdm(total=len(jobs), desc="Tracing", unit="chunk", disable=not progress, leave=False)
    results = executor.run("extract", table, jobs, lambda i, r: bar.update(1))
    bar.close()
    streamlines = []
    for pts, counts in results:
        streamlines.extend(np.split(pts, np.cumsum(counts)[:-1]))
    return streamlines
