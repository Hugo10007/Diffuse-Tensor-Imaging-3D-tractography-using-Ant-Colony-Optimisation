#This script is solely responsible for switching between parallel processing on the CPU, or processing on the GPU if available and supported.
#Uses numpy for CPU, CuPy for GPU. uses data chunking and 'work units' for parallel processing logic.
#An executor is used to run a list of 'jobs' that come and, and returns their results in order. This ensured thread safety and protection against race conditions.
#This script was inspired by my brain tumour detection vision transformer model, which used  a similar execution model to this.

#FOR GPU SUPPORT, INSTALL CUDA (nvidia only): pip install "cupy-cuda12x[ctk]"

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np

from ant_colony_optimisation.ant_colony_optimisation import PheromoneTable, exploring_ants, extract_chunk, join_both_ways, run_ants, split_paths, train_chunk
from data_loading_and_math.measure_water_diffusion_at_voxel import principal_direction_and_fa

INSTALL_HINT = 'pip install "cupy-cuda12x[ctk]"'


#LOGIC FOR CHOOSING THE DEVICE (CPU/GPU) FOUND BELOW:
def gpu_info():
    #Returns 'isusable, description' of the GPU. It also runs a tiny kernel, so it can catch a missing driver or a broken CUDA install.
    try:
        import cupy as cp
    except ImportError:
        return False, f"CuPy is not installed. Install using the following for Nvidia GPU's: ({INSTALL_HINT})"
    except Exception as exc:
        return False, f"CuPy failed to load: {exc}"
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False, "no NVIDIA GPU found, defaulting to CPU."
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        float((cp.arange(16, dtype=cp.float32) ** 2).sum()) #Compiles and runs a basic kernal to test GPU.
        return True, f"{name} ({props['totalGlobalMem'] / 2 ** 30:.0f} GB)" #Returns total GPU memory in gigabytes.
    except Exception as exc:
        return False, f"CUDA is not usable: {exc}"


def default_workers():
    return max((os.cpu_count() or 2), 1) #Returns max thread count.


def make_executor(world, device="auto", n_workers=None, log=None):
    #Makes the executor ('auto' = the GPU if usable, else the CPU).
    #Returns 'executor, description'
    log = log or (lambda *_: None)
    if device in ("auto", "gpu"):
        ok, desc = gpu_info()
        if ok:
            return GPUExecutor(world, log=log), f"GPU: {desc}"
        if device == "gpu":
            log(f"GPU asked for but not usable ({desc}); using the CPU.")
    n = max(int(n_workers or default_workers()), 1)
    executor = ProcessExecutor(world, n) if n > 1 else SerialExecutor(world)
    return executor, f"CPU: {n} process{'es' if n > 1 else ''}"



#LOGIC FOR SINGLE THREAD PROCESSING ON A CPU (legacy stuff, not used anymore unless you have a magic 1 thread 1 core CPU)
def _work(kind):
    return train_chunk if kind == "train" else extract_chunk


class SerialExecutor:
    def __init__(self, world):
        self.world = world

    def run(self, kind, table, jobs, on_result=None):
        results = []
        for i, (seeds, rng_seed) in enumerate(jobs):
            results.append(_work(kind)(self.world, table, seeds, rng_seed))
            if on_result:
                on_result(i, results[-1])
        return results

    def close(self):
        pass


#PARALLEL PROCESSING OF A SINGLE ANT COLONY ON THE CPU (I prefer the term multithreading because it sounds cooler)
_WORKER = {} #Initiated 'the World' for each worker process


def _init_worker(world):
    _WORKER["world"] = world


def _open_shared(name):
    #Creates and removes 'work units', doesnt actually work on them, they're just created and loaded to RAM here.
    #This stuff would actually be more precisely controlled if I was programming in C. I'm tempted to convert everything to C one day to see the perf gains.
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    return shared_memory.SharedMemory(name=name)


def _worker_job(kind, shared, seeds, rng_seed):
    #Runs in a worker thread: rebuilds the pheromone table from shared memory, runs one job.
    #'tau' refers to pheremone. Its the official word but may confuse novices.
    n_bins, tau_initial, tau_min, keys_name, vals_name, n = shared
    blocks = [_open_shared(keys_name), _open_shared(vals_name)] if n else []
    table = PheromoneTable(n_bins, tau_initial, tau_min,
                           np.ndarray((n,), np.int64, buffer=blocks[0].buf) if n else None,
                           np.ndarray((n,), np.float32, buffer=blocks[1].buf) if n else None)
    try:
        return _work(kind)(_WORKER["world"], table, seeds, rng_seed)
    finally:
        del table
        for b in blocks:
            b.close()


def _share(table):
    #Copies the pheromone table into shared memory. Returns 'description for the workers, work blocks'
    n = len(table)
    if n == 0:
        return (table.n_bins, table.tau_initial, table.tau_min, None, None, 0), []
    keys = shared_memory.SharedMemory(create=True, size=table.keys.nbytes)
    vals = shared_memory.SharedMemory(create=True, size=max(table.vals.nbytes, 1))
    np.ndarray((n,), np.int64, buffer=keys.buf)[:] = table.keys
    np.ndarray((n,), np.float32, buffer=vals.buf)[:] = table.vals
    return (table.n_bins, table.tau_initial, table.tau_min, keys.name, vals.name, n), [keys, vals]


class ProcessExecutor:
    #Worker threads class (based on max threads the CPU can support. Mine is 8 threads but fancy people may have more)
    #Creates a pool of the thread count on the cpu. Can  be modified by the user.
    def __init__(self, world, n_workers):
        # 'spawn' works the same on Windows, macOS and Linux
        self.pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"),
                                        initializer=_init_worker, initargs=(world,))

    def run(self, kind, table, jobs, on_result=None):
        #Submits worker jobs to be processed in parallel
        shared, blocks = _share(table)
        try:
            futures = [self.pool.submit(_worker_job, kind, shared, s, r) for s, r in jobs]
            results = []
            for i, f in enumerate(futures):                    # in job order
                results.append(f.result())
                if on_result:
                    on_result(i, results[-1])
            return results
        except BaseException:                                  # e.g. Ctrl+C: stop the workers now
            for p in list(getattr(self.pool, "_processes", {}).values()):
                p.terminate()
            self.pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            for b in blocks:
                b.close()
                try:
                    b.unlink()
                except FileNotFoundError:
                    pass

    def close(self):
        self.pool.shutdown(wait=True)



#GPU PROCESSING IS DONE HERE
class _DeviceRNG:
    #Uniform random numbers made on the device.
    #We need this because NumPy's random number generation is literally too slow for a GPU and we lose perf gains.
    #We need random numbers because the ants choose a random direction in proportion to pheremone, FA and turn direction.

    def __init__(self, xp, seed, dtype):
        self.rs, self.xp, self.dtype = xp.random.RandomState(seed), xp, dtype

    def random(self, n):
        if self.xp is np:                                      # NumPy stand-in (tests)
            return self.rs.random_sample(n).astype(self.dtype)
        return self.rs.random_sample(n, dtype=self.dtype)


def _segment_sums(xp, keys, amounts):
    #Adds up all the pheremone deposits * their strength that landed on the same table entry to be processed on the GPU.
    order = xp.argsort(keys)
    ks = keys[order]
    am = amounts[order].astype(xp.float64)
    if len(ks) == 0:
        return ks, am
    first = xp.concatenate([xp.ones(1, dtype=bool), ks[1:] != ks[:-1]])
    starts = xp.flatnonzero(first)
    ends = xp.concatenate([starts[1:], xp.asarray([len(ks)], dtype=starts.dtype)]) - 1
    cs = xp.cumsum(am)
    before = xp.concatenate([xp.zeros(1, dtype=xp.float64), cs[ends[:-1]]])
    return ks[starts], cs[ends] - before


def _to_host(a):
    #Copies arrays from the GPU back to the CPU. Essentially a converter.
    return a.get() if hasattr(a, "get") else np.asarray(a)


class GPUExecutor:
    #This class is responsible for running the ant colonies jobs on the GPU. This algorithm can walk ~10k ants at once, as opposed to the CPU which is thread-limited.
    #'xp' is CuPy (in Numpy terms).

    def __init__(self, world, xp=None, batch_ants=None, log=None):
        if xp is None:
            import cupy as xp
        self.xp = xp
        self.world = world                                     # CPU copy, for setting up the ants
        self.dworld = world.to_device(xp, np.float32)          # GPU copy, for walking them
        self.log = log or (lambda *_: None)
        if batch_ants is None:                                 # ants per batch: fit about a third of free memory
            batch_ants = 131072
            if xp is not np:
                try:
                    free, _ = xp.cuda.runtime.memGetInfo()
                    batch_ants = int(np.clip(free * 0.35 / 12_000, 16384, 262144))   # ~12 KB per ant
                except Exception:
                    pass
        self.batch_ants = batch_ants

    def run(self, kind, table, jobs, on_result=None):
        #Groups jobs into batches of up to 'batch_ants' ants; one result per batch.
        #We can have much batches due to how GPU's work, as opposed to CPU's which must do jobs one at a time, or in parallel up to n threads on the CPU. 
        dtable = table.to_device(self.xp, np.float32)
        per_seed = self.world.cfg.n_ants_per_seed if kind == "train" else 2
        results, batch, count = [], [], 0
        for i, (seeds, _) in enumerate(jobs):
            n = len(seeds) * per_seed
            if batch and count + n > self.batch_ants:
                results.append(self._batch(kind, dtable, jobs, batch, on_result))
                batch, count = [], 0
            batch.append(i)
            count += n
        if batch:
            results.append(self._batch(kind, dtable, jobs, batch, on_result))
        return results

    def close(self):
        if self.xp is not np:
            try:
                self.xp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def _batch(self, kind, dtable, jobs, idx, on_result):
        #Runs jobs and looks out for small amount of GPU memory, if GPU memory runs out, the batch size gets halved.
        #This was a pain and caused the program to considently crash on high settings.
        fn = self._train if kind == "train" else self._extract
        try:
            res = fn(dtable, [jobs[i] for i in idx])
        except Exception as exc:
            out_of_memory = "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower()  #if out of memory: handles it.
            if not out_of_memory or len(idx) == 1:
                raise
            self.close()
            half = len(idx) // 2
            self.batch_ants = max(self.batch_ants // 2, 4096)
            self.log(f"GPU memory full: batches of {self.batch_ants:,} ants from now on")   #Warning about lack of memory.
            res = _merge(kind, self._batch(kind, dtable, jobs, idx[:half], None),
                         self._batch(kind, dtable, jobs, idx[half:], None))
        for i in idx:
            if on_result:
                on_result(i, None)
        return res

    def _train(self, dtable, jobs):
        #Trains large chunks of ants at once, improving performance/
        xp, w = self.xp, self.dworld
        starts, dirs = [], []
        for seeds, rng_seed in jobs: #Set up exactly as the CPU chunks are
            s, d = exploring_ants(self.world, seeds, np.random.default_rng(rng_seed))
            starts.append(s)
            dirs.append(d)
        rng = _DeviceRNG(xp, int(jobs[0][1]) % (2 ** 31), w.dtype)
        res = run_ants(w, dtable, np.concatenate(starts), np.concatenate(dirs), rng=rng, record_keys=True)
        score = xp.where(res["n_steps"] * w.step_mm < 0.5 * w.cfg.min_length_mm, 0.0, res["quality"])
        amounts = score[res["owner"]]   #every ants step deposits its score
        keep = amounts > 0
        mean_steps = float(res["n_steps"].mean())
        if not bool(keep.any()):
            return np.zeros(0, np.int64), np.zeros(0), mean_steps
        keys, sums = _segment_sums(xp, res["keys"][keep], amounts[keep])
        return _to_host(keys).astype(np.int64), _to_host(sums), mean_steps

    def _extract(self, dtable, jobs):
        #Takes the principle direction at each seed and sends one greedy ant forwards. For multi-ant processing.
        seeds = np.concatenate([s for s, _ in jobs])
        n = len(seeds)
        principal, _ = principal_direction_and_fa(self.world.T, seeds)
        res = run_ants(self.dworld, dtable, np.concatenate([seeds, seeds]), np.concatenate([principal, -principal]),
                       greedy=True, record_paths=True)
        paths = split_paths({"points": _to_host(res["points"]), "n_steps": _to_host(res["n_steps"]).astype(np.int64)})
        return join_both_ways(seeds, paths[:n], paths[n:])


def _merge(kind, a, b):
    #Combines the results of two half-batches incase the users machine ran out of GPU memory.
    #In my case I have 8GB of VRAM and see crashes when processing over ~40k seeds.
    if kind == "train":
        keys, inverse = np.unique(np.concatenate([a[0], b[0]]), return_inverse=True)
        return keys, np.bincount(inverse, weights=np.concatenate([a[1], b[1]])), (a[2] + b[2]) / 2
    return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])