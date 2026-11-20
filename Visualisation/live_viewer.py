"""
A plain 3D window showing the streamlines in the standard DTI colours:
red = left-right, green = front-back, blue = up-down.

Mouse: left-drag rotates, the wheel zooms, shift+drag pans.
"""
from __future__ import annotations

import numpy as np

MAX_POINTS = 10_000_000_000          # draw a random subset of very large tractograms


def show(streamlines, title="ACO Tracker"):
    #Opens the 3D window and blocks it until it is closed.
    import pyvista as pv

    # every other point is plenty for drawing (the tracking steps are half a voxel)
    lines = [s[::2] if len(s) > 4 else s for s in (np.asarray(s, np.float32) for s in streamlines)]
    order = np.random.default_rng(0).permutation(len(lines))
    sizes = np.array([len(lines[i]) for i in order])
    #lines = [lines[i] for i in order[np.cumsum(sizes) <= MAX_POINTS]] or lines[:1] #half resolution
    lines = [np.asarray(s, np.float32) for s in streamlines] #full_resolution
    if not lines:
        print("Nothing to show: no streamlines.")
        return

    points = np.concatenate(lines)
    counts = np.array([len(s) for s in lines])
    # colour of each point: its direction along the streamline, |x| red, |y| green, |z| blue
    tangents = np.concatenate([np.gradient(s, axis=0) if len(s) > 1 else np.ones((1, 3)) for s in lines])
    rgb = np.abs(tangents) / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12)
    rgb = (np.clip(rgb, 0.05, 1.0) * 255).astype(np.uint8)
    # VTK polylines: [n, i0, i1, ..., n, j0, j1, ...]
    cells = np.concatenate([np.concatenate([[n], start + np.arange(n)])
                            for n, start in zip(counts, np.concatenate([[0], np.cumsum(counts)[:-1]]))])
    mesh = pv.PolyData(points, lines=cells)
    mesh.point_data["RGB"] = rgb

    plotter = pv.Plotter(title=title, window_size=(1200, 900))
    plotter.set_background("black")
    plotter.add_mesh(mesh, scalars="RGB", rgb=True, line_width=2, render_lines_as_tubes=True)
    plotter.add_axes(color="white")
    plotter.enable_custom_trackball_style(left="rotate", right="pan", middle="pan", shift_left="pan", control_left="spin")
    plotter.enable_parallel_projection()          # no perspective: panning slides the model without it appearing to turn
    plotter.view_vector((-1, 1, 0.5), viewup=(0, 0, 1))
    plotter.camera.zoom(1.3)
    shown = f"{len(lines):,} of {len(streamlines):,}" if len(lines) < len(streamlines) else f"{len(lines):,}"
    plotter.add_text(f"{shown} streamlines   red = left-right, green = front-back, blue = up-down", font_size=9, color="white")
    plotter.show()
