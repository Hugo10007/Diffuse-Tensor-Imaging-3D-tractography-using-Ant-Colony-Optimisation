#THIS CODE IS RESPONSIBLE FOR TELLING THE ANTS IN THE ACO WHICH WAY THE WATER MOVES AT EACH MRI VOXEL, AND HOW STRONGLY IT MOVES (CALLED 'fractional anistrophy').

"""I did not come up with the diffuse tensor mathematics for this myself. There are standard mathematical formulas (published science) for which water diffusion is calculated,
and even if I did want to come up with this myself, I simply do not possess the know-how in eigendecompositioning to make something as effective
as current research allows. Credits go to Basser & Pierpaoli (1996) on the FA (Fractional Anistrophy) formula.

The voxels are stored as 6 lower triangle components in the format:
Dxx, Dxy, Dyy, Dxz, Dyz, Dzz

Computation is either done using NumPy arrays on the CPU, or CuPy arrays if a Nvidia GPU is available. """  

from __future__ import annotations  #I copied this because I've done some programming using lazy type hints, but I don't think I'll need it.
import sys  #Used for CuPy (GPU-accelerated mathematics)
import numpy as np  #Used for arrays, vectors and matrices

def determine_processor(arr):
    #Determines if we're using the CPU (NumPy) or GPU (CuPy) to process.
    if type(arr).__module__.startswith("cupy"):
        return sys.modules["cupy"]
    else: return np

def _tiny(dtype):
    #Returns a very small number that is NOT 0 so the maths below doesn't attempt to divide by 0 (it took a while to figure this problem out)
    if np.dtype(dtype) == np.float64:
        return 1e-300
    else: return 1e-30


def trilinear_interpolate(volume, positions):
    #Trilinearly interpolation of a scalar (X,Y,Z) or vector (X,Y,Z,C) at continuous voxel coordinates.
    #Trilinear interpolation is used to estimate the value of data at a given point.
    #This is used if an ant is between 2 voxels, we can determine its point by 'blending' the voxels together.
    #This formula was obtained from www.scipy.org
    
    processor = determine_processor(volume)
    
    X, Y, Z = volume.shape[:3]
    pdt = positions.dtype if positions.dtype in (np.float32, np.float64) else np.float64
    hi = processor.asarray([max(X - 1.001, 0), max(Y - 1.001, 0), max(Z - 1.001, 0)], dtype=pdt)
    pos = processor.minimum(processor.maximum(positions.astype(pdt, copy=False), 0), hi)

    x0 = processor.floor(pos[:, 0]).astype(processor.intp); x1 = processor.minimum(x0 + 1, X - 1)
    y0 = processor.floor(pos[:, 1]).astype(processor.intp); y1 = processor.minimum(y0 + 1, Y - 1)
    z0 = processor.floor(pos[:, 2]).astype(processor.intp); z1 = processor.minimum(z0 + 1, Z - 1)
    wdt = volume.dtype if volume.dtype in (np.float32, np.float64) else np.float64
    xd = (pos[:, 0] - x0).astype(wdt); yd = (pos[:, 1] - y0).astype(wdt); zd = (pos[:, 2] - z0).astype(wdt)
    
    if volume.ndim == 4:
        xd = xd[:, None]; yd = yd[:, None]; zd = zd[:, None]
        flat = volume.reshape(-1, volume.shape[3])
    else:
        flat = volume.reshape(-1)
    
    YZ = Y * Z

    def at(xi, yi, zi):          #One flat-index gather is faster than 3-way fancy indexing for quick processing
        return flat[xi * YZ + yi * Z + zi]

    c00 = at(x0, y0, z0) * (1 - xd) + at(x1, y0, z0) * xd
    c01 = at(x0, y0, z1) * (1 - xd) + at(x1, y0, z1) * xd
    c10 = at(x0, y1, z0) * (1 - xd) + at(x1, y1, z0) * xd
    c11 = at(x0, y1, z1) * (1 - xd) + at(x1, y1, z1) * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    return c0 * (1 - zd) + c1 * zd

def eigen(comps):
    #Eigendecomposition occurs here. This is where a 3x3 symmetric matrix gets decomposed into its respective eigenvalues and eigenvectors.
    #Returns eigenvalues in the order of largest first.
    #Trigonometric solution of the characteristic cubic (mathematic formula By Oliver Smith 1961).
    #This is an algorithm which finds 3 eigenvalues of a symmetrix 3x3 matrix
    
    processor = determine_processor(comps)
    if comps.dtype not in (np.float32, np.float64):
        comps = comps.astype(np.float64)
    tiny = _tiny(comps.dtype)
    a, d, b, e, f, c = (comps[:, i] for i in range(6))

    p1 = d * d + e * e + f * f
    q = (a + b + c) / 3.0
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1
    p = processor.sqrt(processor.maximum(p2 / 6.0, tiny))

    Baa = (a - q) / p; Bbb = (b - q) / p; Bcc = (c - q) / p
    Bab = d / p; Bac = e / p; Bbc = f / p
    detB = (Baa * (Bbb * Bcc - Bbc * Bbc) - Bab * (Bab * Bcc - Bbc * Bac) + Bac * (Bab * Bbc - Bbb * Bac))
    phi = processor.arccos(processor.clip(detB / 2.0, -1.0, 1.0)) / 3.0

    eig1 = q + 2 * p * processor.cos(phi)
    eig3 = q + 2 * p * processor.cos(phi + 2 * np.pi / 3)
    eig2 = 3 * q - eig1 - eig3
    evals = processor.stack([eig1, eig2, eig3], axis=1)

    scale = processor.abs(a) + processor.abs(b) + processor.abs(c) + tiny
    isotropic = p1 < 1e-18 * scale * scale
    diag_sorted = processor.sort(processor.stack([a, b, c], axis=1), axis=1)[:, ::-1]
    evals = processor.where(isotropic[:, None], diag_sorted, evals)

    #Principal eigenvector: the largest cross product of two rows: D - eig1 I
    r0 = processor.stack([a - eig1, d, e], axis=1)
    r1 = processor.stack([d, b - eig1, f], axis=1)
    r2 = processor.stack([e, f, c - eig1], axis=1)
    c01 = processor.cross(r0, r1); c02 = processor.cross(r0, r2); c12 = processor.cross(r1, r2)
    stacked = processor.stack([c01, c02, c12], axis=1)
    norms = processor.einsum("nkj,nkj->nk", stacked, stacked)
    vec = stacked[processor.arange(len(comps)), processor.argmax(norms, axis=1)]
    norm = processor.linalg.norm(vec, axis=1, keepdims=True)
    safe = norm[:, 0] > 1e-30
    fallback = processor.asarray([0.0, 0.0, 1.0], dtype=vec.dtype)
    principal = processor.where(safe[:, None], vec / processor.where(safe[:, None], norm, 1.0), fallback)
    return evals, principal

def fa_from_eigenvalues(evals):
    #Fractional Anisotropy (N,) from eigenvalues. This determines the actual strength water is moving in a direction by determining how unequal the 3 eigenvalues are.
    #Used by ants to determine where fibers are. Ranges from 0 (no motion or random motion) to 1 (full motion in 1 direction).
    processor = determine_processor(evals)
    l1, l2, l3 = evals[:, 0], evals[:, 1], evals[:, 2]
    num = (l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2
    den = l1 ** 2 + l2 ** 2 + l3 ** 2
    ok = den > 0
    fa = processor.where(ok, processor.sqrt(0.5 * num / processor.where(ok, den, 1.0)), 0.0)
    return processor.clip(fa, 0.0, 1.0)

def local_tensor(tensor_lt, positions):
    #Determines a tensors if its between 2 voxels: components (N,6), eigenvalues (N,3).
    #This works by using the trilinear interpolation algorithm to 'blend' the voxels together.
    dt = np.float32 if positions.dtype == np.float32 else np.float64
    comps = trilinear_interpolate(tensor_lt, positions).astype(dt, copy=False)
    evals, principal = eigen(comps)
    return comps, evals, principal, fa_from_eigenvalues(evals)

def principal_direction_and_fa(tensor_lt, positions):
    #This determines the direction of which water is flowing (NOT the intensity)
    _, _, principal, fa = local_tensor(tensor_lt, positions)
    return principal, fa

def tensor_times(comps, v):
    #Takes a direction and returns where the tensor is pushing it based on the 3x3 matrix values. built on by apparent_diffusivity().
    processor = determine_processor(comps)
    xx, xy, yy, xz, yz, zz = (comps[:, i] for i in range(6))
    return processor.stack([xx * v[:, 0] + xy * v[:, 1] + xz * v[:, 2],
                     xy * v[:, 0] + yy * v[:, 1] + yz * v[:, 2],
                     xz * v[:, 0] + yz * v[:, 1] + zz * v[:, 2]], axis=1)

def apparent_diffusivity(comps, d):
    #Measures how freely water diffuses along unit direction d.
    return determine_processor(comps).einsum("ij,ij->i", tensor_times(comps, d), d)