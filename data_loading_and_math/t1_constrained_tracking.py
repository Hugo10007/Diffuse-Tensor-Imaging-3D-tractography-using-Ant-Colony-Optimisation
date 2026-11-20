#Constrains tracking to only the white matter of the patients brain IF patient also had a T1 MRI scan and its inside the loaded files metadata.
#This improves accuracy due to t1 scans explicitly imaging white matter and gray matter within a patient at a much higher rsolution than standard DTI MRI imaging.
#If you are struggling to load your t1 scan into the DTI scans metadata, ask Hugo for help.
#Uses wm_threshold to determine what is and isn't white matter. A streamline is only kept if both ends reach gray matter.
#The colony itself remains unchanged, only where they can walk changes.

from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import numpy as np

#SETTINGS OF THE T1 CONSTRAINTS. THESE CANT BE MODIFIED VIA THE CLI (STATIC VARIABLES).
@dataclass
class AnatomyConfig:
    enabled: bool = True #Uses the tissue maps whenever the ACO-ready MRI file has them
    wm_threshold: float = 0.25 #Track where the white-matter fraction is at least this
    fa_floor: float = 0.05 #Stops the fiber when from choosing a direction fractional anistrophy below this value
    gm_threshold: float = 0.3 #A voxel (outside the white-matter mask) is considered grey matter from this fraction
    end_reach_vox: int = 1 #Ends must be within this many voxels of grey matter
    edge_reach_vox: int = 2 #Fiber can also end if on the edge of the brain


def _binary_dialation(m):
    #Binary dilation. This is where we grow the true/false mask around white matter, gray matter, and the brain mask.
    #It works by allowing each voxel to make its neighboring voxel true too. (left, right, up, down, front, back)
    #This is needed so we can expand the mask where the nerve fibers are until it reaches gray matter, then we know how far the ants can walk.
    out = m.copy()
    for ax in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        out[tuple(lo)] |= m[tuple(hi)]
        out[tuple(hi)] |= m[tuple(lo)]
    return out


def dilate(m, iterations=1):
    #Equivalent of: scipy.ndimage.binary_dilation(m, iterations=n) (default structure).
    m = np.asarray(m, bool)
    for _ in range(int(iterations)):
        m = _binary_dialation(m)
    return m


def erode(m, iterations=1):
    #Equivalent of scipy.ndimage.binary_erosion(m, iterations=n) (default structure).
    #Used to find the brain through binary erosion.
    m = np.asarray(m, bool)
    for _ in range(int(iterations)):
        p = np.pad(m, 1, constant_values=False)
        m = ~_binary_dialation(~p)[1:-1, 1:-1, 1:-1]
    return m


def available(ready) -> bool:
    #Checks if t1 data is available
    return getattr(ready, "tissue", None) is not None


def active(ready, cfg: AnatomyConfig) -> bool:
    #Valid settings and t1 data
    return bool(cfg is not None and cfg.enabled and available(ready))


def white_matter_mask(ready, cfg: AnatomyConfig):
    #Creates white matter mask of white matter tissue (where fibers run)
    return ready.brain_mask & (ready.tissue[..., 0] >= cfg.wm_threshold)


def grey_matter_mask(ready, cfg: AnatomyConfig):
    #Creates gray matter mask of white matter tissue (where fibers end)
    return ready.brain_mask & ~white_matter_mask(ready, cfg) & (ready.tissue[..., 1] >= cfg.gm_threshold)


def constrain(ready, aco_cfg, cfg: AnatomyConfig):
    #Returns 'ready, aco_cfg' to track with: the white-matter tracking mask and the FA floor in place of the stop-FA threshold.
    wm = white_matter_mask(ready, cfg)
    if not wm.any():
        raise ValueError("The white-matter map is empty; check the T1 used in DTI Prep.")
    r = dataclasses.replace(ready, tracking_mask=wm)
    a = dataclasses.replace(aco_cfg, stop_fa_threshold=float(cfg.fa_floor))
    return r, a


def allowed_ends(ready, cfg: AnatomyConfig):
    #Where a streamline is allowed to end
    gm_near = dilate(grey_matter_mask(ready, cfg), max(int(cfg.end_reach_vox), 0))
    edge = ~erode(ready.brain_mask, max(int(cfg.edge_reach_vox), 1))
    return gm_near | edge


def end_filter(streamlines_vox, ready, cfg: AnatomyConfig, ends_ok=None):
    #Indices of the streamlines where its 2 ends are in valid gray matter
    if not len(streamlines_vox):
        return np.zeros(0, int)
    ok = allowed_ends(ready, cfg) if ends_ok is None else ends_ok
    hi = np.array(ok.shape) - 1
    ends = np.array([[s[0], s[-1]] for s in streamlines_vox], np.float64) #(n, 2, 3)
    ijk = np.clip(np.rint(ends).astype(np.int64), 0, hi)
    good = ok[ijk[..., 0], ijk[..., 1], ijk[..., 2]].all(axis=1)
    return np.flatnonzero(good)


def describe(ready, cfg: AnatomyConfig):
    #Small explanation for the user.
    wm = white_matter_mask(ready, cfg)
    gm = grey_matter_mask(ready, cfg)
    return (f"white matter {int(wm.sum()):,} voxels (fraction >= {cfg.wm_threshold:g}), grey matter "
            f"{int(gm.sum()):,} voxels; FA floor {cfg.fa_floor:g}; ends must reach grey matter or the brain's edge")