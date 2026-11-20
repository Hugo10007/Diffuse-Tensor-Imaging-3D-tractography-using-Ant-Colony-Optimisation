#Traces the paths from raw ant paths to a clean tractogram: length limits and smoothing.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import nibabel as nib



#SETTINGS OF THE FIBER FILTERING (not accessible via the CLI as of now.)
@dataclass
class CleanupConfig:
    enabled: bool = True                 # group streamlines by shape; remove small, isolated groups (wrong turns)
    min_cluster_size: int = 10           # smaller groups are removed



#LIMITING LENGTH OF STREAMLINES TO VALID LENGTHS & APPLYING LIGHT SMOOTHING 
def lengths_mm(streamlines_vox, voxel_size):
    vs = np.asarray(voxel_size, dtype=np.float64)
    return np.array([np.sum(np.linalg.norm(np.diff(s, axis=0) * vs, axis=1)) if len(s) > 1 else 0.0
                     for s in streamlines_vox])


def filter_by_length(streamlines_vox, voxel_size, min_mm, max_mm):
    L = lengths_mm(streamlines_vox, voxel_size)
    keep = (L >= min_mm) & (L <= max_mm)
    return [s for s, k in zip(streamlines_vox, keep) if k], L[keep]


def smooth(streamline, window=5):
    #Moving-average smoothing along the path. endpoints remain where they were originally.
    n = len(streamline)
    w = int(window) | 1  # force odd
    if n < 3 or w < 3:
        return streamline
    w = min(w, n if n % 2 else n - 1)
    h = w // 2
    padded = np.pad(streamline.astype(np.float64), ((h, h), (0, 0)), mode="edge")
    c = np.cumsum(padded, axis=0)
    c = np.vstack([np.zeros((1, 3)), c])
    out = (c[w:] - c[:-w]) / w
    out[0], out[-1] = streamline[0], streamline[-1]
    return out.astype(streamline.dtype)


def apply_affine(points_vox, affine):
    return nib.affines.apply_affine(affine, points_vox)


#Sorting BY LENGTH (removing fibers less than 2mm)
def resample(streamline, n_points=12):
    #Used to convert ALL streamlines (whether made with 60 or 200 points) and converts it to 12 points.
    #This is so we can compare streamlines that are similar, so we can favour them.
    s = np.asarray(streamline, dtype=np.float64)
    if len(s) < 2:
        return np.repeat(s[:1], n_points, axis=0)
    seg = np.sqrt(((s[1:] - s[:-1]) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return np.repeat(s[:1], n_points, axis=0)
    t = np.linspace(0.0, cum[-1], n_points)
    return np.stack([np.interp(t, cum, s[:, k]) for k in range(3)], axis=1)

def cluster_and_filter(streamlines_mm, cfg: CleanupConfig, log=None):
    #Filters fibers that are anatomically too small. (less than 2mm)
    n = len(streamlines_mm)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int), 0
    groups, sizes = quickbundles(streamlines_mm, cfg.cluster_threshold_mm)
    order = np.argsort(-sizes, kind="stable")
    new_label = np.full(len(sizes), -1, dtype=int)
    label = 0
    for gi in order:
        if sizes[gi] < cfg.min_cluster_size:
            continue
        new_label[gi] = label
        label += 1
    labels = new_label[groups]
    keep = np.flatnonzero(labels >= 0)
    if log:
        log(f"  bundle filter: {len(sizes):,} groups; kept {label:,} bundles of >= {cfg.min_cluster_size} "
            f"streamlines ({len(keep):,} of {n:,} streamlines, {100.0 * len(keep) / n:.0f}%)")
    return keep, labels[keep], label

def quickbundles(streamlines, threshold_mm, n_points=12):
    #Returns (label per streamline, size per group). Groups are numbered in the order they were created.
    #Not my work, inspired by Garyfallidis et al. 2012 (view references in paper)
    n = len(streamlines)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    feats = np.empty((n, n_points, 3), np.float64)
    for i, s in enumerate(streamlines):
        feats[i] = resample(s, n_points)
    means = feats.mean(axis=1)
    thr = float(threshold_mm)

    cap = 1024
    cent = np.empty((cap, n_points, 3))
    cmean = np.empty((cap, 3))
    sizes = np.zeros(cap, np.int64)
    labels = np.empty(n, np.int64)
    k_used = 0
    for i in range(n):
        f = feats[i]
        if k_used:
            lb = np.sqrt(((cmean[:k_used] - means[i]) ** 2).sum(axis=1))
            cand = np.flatnonzero(lb < thr)
            if len(cand):
                c = cent[cand]
                d_fwd = np.sqrt(((c - f) ** 2).sum(axis=2)).mean(axis=1)
                d_rev = np.sqrt(((c - f[::-1]) ** 2).sum(axis=2)).mean(axis=1)
                j_fwd, j_rev = int(np.argmin(d_fwd)), int(np.argmin(d_rev))
                if d_rev[j_rev] < d_fwd[j_fwd]:
                    j, d, add = j_rev, d_rev[j_rev], f[::-1]
                else:
                    j, d, add = j_fwd, d_fwd[j_fwd], f
                if d < thr:
                    k = cand[j]
                    m = sizes[k]
                    cent[k] = (cent[k] * m + add) / (m + 1)
                    cmean[k] = cent[k].mean(axis=0)
                    sizes[k] = m + 1
                    labels[i] = k
                    continue
        if k_used == cap:
            cap *= 2
            cent = np.concatenate([cent, np.empty_like(cent)])
            cmean = np.concatenate([cmean, np.empty_like(cmean)])
            sizes = np.concatenate([sizes, np.zeros_like(sizes)])
        cent[k_used] = f
        cmean[k_used] = means[i]
        sizes[k_used] = 1
        labels[i] = k_used
        k_used += 1
    return labels, sizes[:k_used].copy()