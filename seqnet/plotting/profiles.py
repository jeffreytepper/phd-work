"""Views of connectivity profiles that change along the sequence (see ``seqnet.analysis.neuron_profiles``).

Every function draws on a given Axes and takes the profiles as arrays, so the notebooks only load data and lay out
figures.  Positions are sequence positions of neurons (any origin); ``pulse`` marks the pulse neuron's position.
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np

__all__ = ["position_colors", "overlay", "ridgeline", "position_map", "mark_pulse"]


def position_colors(positions: np.ndarray, cmap: str = "turbo") -> tuple[np.ndarray, mpl.cm.ScalarMappable]:
    """Colours for sequence positions and the mappable for a colorbar."""
    norm = mpl.colors.Normalize(float(np.min(positions)), float(np.max(positions)))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return sm.to_rgba(positions), sm


def overlay(ax, d, profiles, positions, pulse=None, control=None, cmap="turbo", lw=1.0):
    """All binned profiles on one axis, coloured by position; the bin holding the pulse in black; ``control`` (the
    control's mean profile) dashed grey.  Returns the colour mappable."""
    cols, sm = position_colors(positions, cmap)
    if control is not None:
        ax.plot(d, control, color="0.4", ls="--", lw=1.5, zorder=3)
    for p, c in zip(profiles, cols):
        ax.plot(d, p, color=c, lw=lw, alpha=0.85)
    if pulse is not None:
        i = int(np.argmin(np.abs(np.asarray(positions) - pulse)))
        ax.plot(d, profiles[i], color="k", lw=2.0, zorder=4)
    ax.axvline(0, color="k", lw=0.4)
    return sm


def ridgeline(ax, d, profiles, positions, pulse=None, control=None, spacing=None, cmap="turbo", fill=True):
    """Profiles stacked bottom (first position) to top, coloured by position, each over its control (grey).

    ``control``: (n_bins, D) control profiles of the same bins.  ``spacing`` defaults to the profiles' range.
    """
    profiles = np.asarray(profiles)
    spacing = spacing or 0.6 * float(np.ptp(profiles))
    cols, _ = position_colors(positions, cmap)
    ipulse = None if pulse is None else int(np.argmin(np.abs(np.asarray(positions) - pulse)))
    for i in range(len(profiles) - 1, -1, -1):  # draw top rows first so lower rows overlap them
        y0 = i * spacing
        if control is not None:
            ax.plot(d, control[i] + y0, color="0.55", lw=0.8)
        ax.plot(d, profiles[i] + y0, color="k" if i == ipulse else cols[i], lw=1.6 if i == ipulse else 1.0)
        if fill:
            ax.fill_between(d, y0, profiles[i] + y0, color=cols[i], alpha=0.25, lw=0)
    ax.set_yticks(np.arange(len(profiles)) * spacing)
    ax.set_yticklabels([f"{p:.0f}" for p in positions], fontsize=6)
    ax.axvline(0, color="k", lw=0.4)


def position_map(ax, values, positions, x, pulse=None, cmap="RdBu_r", vlim=None, label=None):
    """Image of values[position, x] (positions on y), symmetric colour range; returns the image."""
    v = vlim if vlim is not None else float(np.nanmax(np.abs(values)))
    im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap, vmin=-v, vmax=v, interpolation="nearest",
                   extent=[x[0] - 0.5, x[-1] + 0.5, positions[0], positions[-1]])
    if pulse is not None:
        mark_pulse(ax, pulse, "y")
    if label:
        ax.set_title(label, fontsize=9)
    return im


def mark_pulse(ax, pos, axis="x", **kw):
    """Dashed line at the pulse neuron's position."""
    kw = {"color": "k", "ls": "--", "lw": 0.7, **kw}
    (ax.axvline if axis == "x" else ax.axhline)(pos, **kw)
