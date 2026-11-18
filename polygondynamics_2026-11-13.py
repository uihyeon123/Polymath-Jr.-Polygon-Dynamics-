#!/usr/bin/env python3
import importlib.util
import os
import subprocess
import sys


def _under_streamlit() -> bool:
    """True when this script is being executed by the Streamlit runtime."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    try:
        
        return get_script_run_ctx(suppress_warning=True) is not None
    except TypeError:
        return get_script_run_ctx() is not None


def _skip_first_run_prompt() -> None:
    """On its very first run Streamlit asks for an email address and waits on
    stdin, which stalls the launcher (some Run consoles have no keyboard).
    Write the same empty-email file Streamlit writes when you press Enter."""
    path = os.path.join(os.path.expanduser("~"), ".streamlit", "credentials.toml")
    if os.path.exists(path):
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write('[general]\nemail = ""\n')
    except OSError:
        pass  


def _relaunch_under_streamlit() -> None:
    """Re-run this file through `streamlit run` and wait for it to finish."""
    if importlib.util.find_spec("streamlit") is None:
        sys.exit(
            "Streamlit is not installed in this interpreter.\n"
            f"Install it with:  {sys.executable} -m pip install streamlit numpy matplotlib"
        )

    _skip_first_run_prompt()

    script = os.path.abspath(__file__)
    cmd = [
        sys.executable, "-m", "streamlit", "run", script,
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    print("Starting the app in your browser... press Ctrl+C here to stop it.")
    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        sys.exit(0)


import inspect
import io

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib import patheffects
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


FONT_SCALE = 1.4


def fs(pt):
    """A font size in points, scaled by FONT_SCALE."""
    return round(pt * FONT_SCALE, 1)


PAPER_STYLE = {
    
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": fs(10),
    "axes.titlesize": fs(11),
    "axes.labelsize": fs(10.5),
    "xtick.labelsize": fs(9),
    "ytick.labelsize": fs(9),
    "legend.fontsize": fs(9),
    
    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.2",
    "axes.labelcolor": "0.1",
    "xtick.color": "0.2",
    "ytick.color": "0.2",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.5,
    
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,     
    "ps.fonttype": 42,
}
matplotlib.rcParams.update(PAPER_STYLE)



OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#000000", "#F0E442"]

POLY_FILL = "#E9EEF5"      
POLY_EDGE = "0.15"
DIAG_COLOR = "0.45"
FOLD_LINE = "#C0392B"      
ORBIT_GREY = "0.55"
ANIM_DPI = 100             
STILL_DPI = 200            

MIME = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}


def _full_width(fn):
    """Streamlit renamed use_container_width to width; support both."""
    if "width" in inspect.signature(fn).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


WIDE_BUTTON = _full_width(st.button)
WIDE_DOWNLOAD = _full_width(st.download_button)
WIDE_PLOT = _full_width(st.pyplot)






def reflect_across_line(z, a, b):
    """Reflect z across the line through a and b."""
    d = b - a
    return np.conj(z - a) * d / np.conj(d) + a


def reflect_across_perp_bisector(z, a, b):
    """Reflect z across the perpendicular bisector of the segment ab.

    Equivalent to reflecting across the line through (a+b)/2 in direction
    i*(b-a); this map sends a to b and b to a.
    """
    m = 0.5 * (a + b)
    d = b - a
    return -np.conj(z - m) * d / np.conj(d) + m






def fold(v):
    """Reflect v0 across the diagonal joining its neighbours, then relabel."""
    p = reflect_across_line(v[0], v[1], v[-1])
    return np.concatenate([v[1:], [p]])


def fold_reverse(v):
    """Same reflection, labels advancing the other way round the polygon."""
    p = reflect_across_line(v[0], v[1], v[-1])
    return np.concatenate([[v[-1], p], v[1:-1]])


def recut(v):
    """Reflect v0 across the perpendicular bisector of that diagonal."""
    p = reflect_across_perp_bisector(v[0], v[1], v[-1])
    return np.concatenate([v[1:], [p]])


def snap(v):
    """Move v0 to the midpoint of its two neighbours."""
    p = 0.5 * (v[1] + v[-1])
    return np.concatenate([v[1:], [p]])


def fold_then_recut(v):
    """One diagonal reflection followed by one recut."""
    return recut(fold(v))


FOLD_FUNCS = {
    "Diagonal Reflection": fold,
    "Diagonal Reflection in Reverse Order": fold_reverse,
    "Snap": snap,
    "Recut": recut,
    "Fold then Recut": fold_then_recut,
}

FOLD_NOTES = {
    "Diagonal Reflection":
        "Reflects v0 across the line through v1 and v{last}, then relabels so the "
        "reflected point lands at the end.",
    "Diagonal Reflection in Reverse Order":
        "The same reflection, but the labels advance the other way round the polygon.",
    "Snap":
        "Moves v0 to the midpoint of v1 and v{last}. Areas shrink, so orbits collapse.",
    "Recut":
        "Reflects v0 across the perpendicular bisector of the segment from v1 to v{last}.",
    "Fold then Recut":
        "One diagonal reflection immediately followed by one recut.",
}


def centered(op):
    """Wrap an operation so the centroid sits at the origin after each step."""
    def step(v):
        w = op(v)
        return w - w.mean()
    return step






def iterate(v, steps, op_name, center=False):
    """Return (frames, hit_degenerate). Each frame is a length-n complex array."""
    op = FOLD_FUNCS[op_name]
    if center:
        op = centered(op)

    v = np.asarray(v, dtype=complex)
    if center:
        v = v - v.mean()

    frames = [v]
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(steps):
            v = op(v)
            if not np.all(np.isfinite(v)):
                return frames, True
            frames.append(v)
    return frames, False


def all_points(frames):
    """Flatten frames into x and y arrays, keeping vertex index grouping."""
    stack = np.array(frames)                 
    return stack.real, stack.imag


def default_vertices(n):
    """An irregular convex n-gon. Symmetric polygons fold onto themselves and
    stall, so the angles and radii are both deliberately off-balance."""
    if n == 4:
        return np.array([-1 - 0.5j, 1 - 0.3j, 1 + 0.6j, -1 + 0.4j])
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    ang = ang + 0.13 * np.sin(2 * ang + 1.3)
    r = 1.0 + 0.18 * np.cos(3 * ang + 0.9)
    return np.round(r * np.exp(1j * ang), 3)


def random_vertices(n, rng):
    """Random convex-ish polygon: sorted angles with jittered radii."""
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(0.6, 1.3, n)
    return np.round(r * np.exp(1j * ang), 3)


def diagonal_pairs(n, gaps=None):
    """Distinct diagonals (i, j), shortest span first.

    `gaps` restricts the result to certain spans: gap 2 joins vertices one
    apart ("skip one"), gap 3 skips two, and so on up to n // 2. A hexagon has
    six gap-2 diagonals and three gap-3 ones through the middle; passing
    gaps=[2] keeps just the short family, which is far easier to read.
    """
    available = range(2, n // 2 + 1)
    wanted = available if gaps is None else [g for g in gaps if g in available]
    pairs = []
    for gap in wanted:
        for i in range(n):
            key = (min(i, (i + gap) % n), max(i, (i + gap) % n))
            if key not in pairs:
                pairs.append(key)
    return pairs


def gap_families(n):
    """The distinct diagonal spans a polygon with n vertices has."""
    return list(range(2, n // 2 + 1))






def vertex_colors(n):
    return [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n)]


def series_colors(k):
    """Colours for k tracked diagonals: the vertex palette while it lasts,
    then an evenly spaced viridis ramp when there are too many."""
    if k <= 7:
        return OKABE_ITO[:k]
    return list(plt.cm.viridis(np.linspace(0.0, 0.9, k)))


def vlabel(i):
    """Math-mode vertex label, e.g. $v_{3}$."""
    return rf"$v_{{{i}}}$"


def axes_cross(ax):
    """Faint dashed lines through the origin instead of heavy black ones."""
    ax.axhline(0, color="0.6", linewidth=0.6, linestyle=(0, (4, 3)), zorder=0.5)
    ax.axvline(0, color="0.6", linewidth=0.6, linestyle=(0, (4, 3)), zorder=0.5)


def finish_axes(ax, xlim, ylim, title=None, axis_labels=True):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    if axis_labels:
        ax.set_xlabel(r"$\mathrm{Re}\,z$")
        ax.set_ylabel(r"$\mathrm{Im}\,z$")
    if title:
        ax.set_title(title, pad=8)


def draw_polygon(ax, v, colors, diagonals=None, label=False, s=30):
    x = np.append(v.real, v[0].real)
    y = np.append(v.imag, v[0].imag)
    ax.fill(x, y, color=POLY_FILL, alpha=0.9, zorder=1)
    ax.plot(x, y, color=POLY_EDGE, linewidth=1.3, solid_joinstyle="round",
            zorder=2)

    for i, j in (diagonals or []):
        ax.plot([v[i].real, v[j].real], [v[i].imag, v[j].imag],
                color=DIAG_COLOR, linestyle=(0, (3, 2)), linewidth=0.75,
                zorder=2)

    ax.scatter(v.real, v.imag, color=colors, s=s, zorder=5,
               edgecolors="white", linewidths=0.7)

    if label:
        halo = [patheffects.withStroke(linewidth=3, foreground="white")]
        for i, z in enumerate(v):
            ax.annotate(vlabel(i), (z.real, z.imag), textcoords="offset points",
                        xytext=(7, 6), fontsize=fs(10.5), color="black", zorder=6,
                        path_effects=halo)


def preview_figure(v):
    """Small static picture of the starting polygon, with the fold line marked."""
    n = len(v)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    
    draw_polygon(ax, v, vertex_colors(n), diagonals=diagonal_pairs(n, [2]),
                 label=True, s=30)
    
    ax.plot([v[1].real, v[-1].real], [v[1].imag, v[-1].imag],
            color=FOLD_LINE, linewidth=1.8, alpha=0.95, zorder=4,
            label=rf"fold line $v_1 v_{{{n - 1}}}$")
    axes_cross(ax)
    lim = max(1.0, np.abs(v).max() * 1.25)
    finish_axes(ax, (-lim, lim), (-lim, lim))
    ax.legend(loc="lower right", fontsize=fs(8.5), handlelength=1.6)
    fig.tight_layout()
    return fig






def render_bytes(fig, formats=("png", "pdf")):
    """Render a figure to in-memory files for the download buttons."""
    out = {}
    for fmt in formats:
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=300 if fmt == "png" else None)
        out[fmt] = buf.getvalue()
    return out


def download_row(files, basename, key):
    cols = st.columns([1, 1, 3])
    for col, (fmt, data) in zip(cols, files.items()):
        with col:
            st.download_button(f"Download {fmt.upper()}", data,
                               file_name=f"{basename}.{fmt}", mime=MIME[fmt],
                               key=f"{key}_{fmt}", **WIDE_DOWNLOAD)


def remember_figure(key, fig, **extra):
    """Keep a rendered figure across reruns so a download click does not
    blank the page. Closes whatever was stored before."""
    old = st.session_state.get(key)
    if old is not None:
        plt.close(old["fig"])
    st.session_state[key] = {"fig": fig, "files": render_bytes(fig), **extra}


def show_stored_figure(key, basename, width_cols=(1, 2.5, 1)):
    res = st.session_state.get(key)
    if res is None:
        return None
    _, mid, _ = st.columns(list(width_cols))
    with mid:
        st.pyplot(res["fig"], **WIDE_PLOT)
        download_row(res["files"], basename, key)
    return res






def plot_orbit(v, steps, plotsize, pointsize, op_name, color_by_vertex, center,
               show_title=True):
    """plotsize=None fits the axes to the data instead of using a fixed window."""
    frames, degenerate = iterate(v, steps, op_name, center=center)
    xs, ys = all_points(frames)
    n = xs.shape[1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.1)

    if color_by_vertex:
        colors = vertex_colors(n)
        for i in range(n):
            ax.scatter(xs[:, i], ys[:, i], color=colors[i], s=pointsize,
                       alpha=0.6, linewidths=0, rasterized=True, label=vlabel(i))
        leg = ax.legend(loc="upper right", markerscale=max(2.0, 10.0 / pointsize),
                        scatterpoints=1)
        for h in leg.legend_handles:
            h.set_alpha(1.0)
    else:
        ax.scatter(xs.ravel(), ys.ravel(), color="0.1", s=pointsize, alpha=0.6,
                   linewidths=0, rasterized=True)

    axes_cross(ax)

    if plotsize is None:
        pad = 0.05 * max(np.ptp(xs), np.ptp(ys)) + 0.05
        xlim = (xs.min() - pad, xs.max() + pad)
        ylim = (ys.min() - pad, ys.max() + pad)
        clipped = 0.0
    else:
        xlim = ylim = (-plotsize, plotsize)
        clipped = float(np.mean((np.abs(xs) > plotsize) | (np.abs(ys) > plotsize)))

    title = (f"Orbit of a {n}-gon, {len(frames) - 1} iterations"
             if show_title else None)
    finish_axes(ax, xlim, ylim, title)
    return fig, degenerate, clipped






def _folding_builder(v, steps, plotsize, pointsize, show_orbit, orbit_steps,
                     orbit_alpha, op_name, center, label, show_diagonals, dpi):
    """Build the figure and a frame-drawing function shared by the animation
    and the single-frame export."""
    n = len(v)
    colors = vertex_colors(n)
    diagonals = diagonal_pairs(n) if show_diagonals else None

    orbit_x = orbit_y = None
    if show_orbit:
        o_frames, _ = iterate(v, orbit_steps, op_name, center=center)
        ox, oy = all_points(o_frames)
        orbit_x, orbit_y = ox.ravel(), oy.ravel()

    frames, degenerate = iterate(v, steps, op_name, center=center)
    suffix = " (centred)" if center else ""

    if plotsize is None:
        stack = np.array(frames)
        px = list(stack.real.ravel()) + (list(orbit_x) if show_orbit else [])
        py = list(stack.imag.ravel()) + (list(orbit_y) if show_orbit else [])
        pad = 0.05 * max(np.ptp(px), np.ptp(py)) + 0.05
        xlim = (min(px) - pad, max(px) + pad)
        ylim = (min(py) - pad, max(py) + pad)
    else:
        xlim = ylim = (-plotsize, plotsize)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=dpi)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.1)

    def update(i, show_title=True):
        ax.clear()
        if show_orbit:
            ax.scatter(orbit_x, orbit_y, color=ORBIT_GREY, s=pointsize,
                       alpha=orbit_alpha, linewidths=0, rasterized=True)
        axes_cross(ax)
        draw_polygon(ax, frames[i], colors, diagonals, label, s=30)
        finish_axes(ax, xlim, ylim,
                    f"Iteration {i}{suffix}" if show_title else None)

    return fig, update, len(frames), degenerate


def animate_folding(v, steps, interval_ms, plotsize, pointsize, show_orbit,
                    orbit_steps, orbit_alpha, op_name, center=False,
                    label=True, show_diagonals=False):
    fig, update, n_frames, degenerate = _folding_builder(
        v, steps, plotsize, pointsize, show_orbit, orbit_steps, orbit_alpha,
        op_name, center, label, show_diagonals, ANIM_DPI)
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval_ms, repeat=True)
    html = anim.to_jshtml()
    plt.close(fig)
    return html, degenerate


def folding_still(v, steps, k, plotsize, pointsize, show_orbit, orbit_steps,
                  orbit_alpha, op_name, center, label, show_diagonals,
                  show_title):
    """A single iteration drawn at print resolution."""
    fig, update, n_frames, degenerate = _folding_builder(
        v, steps, plotsize, pointsize, show_orbit, orbit_steps, orbit_alpha,
        op_name, center, label, show_diagonals, STILL_DPI)
    k = min(k, n_frames - 1)
    update(k, show_title)
    return fig, degenerate, k






def diagonal_lengths(frames, pairs):
    """Squared length of each diagonal at every iteration -> (n_frames, n_pairs)."""
    stack = np.array(frames)
    return np.array([np.abs(stack[:, i] - stack[:, j]) ** 2 for i, j in pairs]).T


def _diagonal_dynamics_builder(v, steps, quad_window, op_name, pair_a, pair_b,
                               right_mode, pairs, fade_steps, label, dpi):
    n = len(v)
    colors = vertex_colors(n)
    pairs = pairs if pairs is not None else diagonal_pairs(n)

    frames, degenerate = iterate(v, steps, op_name, center=True)
    lengths = diagonal_lengths(frames, pairs)
    n_frames = len(frames)

    ia, ib = pairs.index(pair_a), pairs.index(pair_b)
    trace_x, trace_y = lengths[:, ia], lengths[:, ib]

    pad = 1.0
    xlim = (trace_x.min() - pad, trace_x.max() + pad)
    ylim = (trace_y.min() - pad, trace_y.max() + pad)

    segs = np.stack([np.column_stack([trace_x[:-1], trace_y[:-1]]),
                     np.column_stack([trace_x[1:], trace_y[1:]])], axis=1)

    fig_h = 5.2
    fig, (ax_poly, ax_right) = plt.subplots(1, 2, figsize=(10.5, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.14, wspace=0.32)
    line_colors = series_colors(len(pairs))

    def fading_collection(i):
        if i < 1:
            return None
        k = np.arange(i)
        alpha = np.full(i, 0.25)
        recent = k >= i - fade_steps
        alpha[recent] = 0.25 + 0.65 * (k[recent] - (i - fade_steps)) / max(fade_steps, 1)
        rgba = np.zeros((i, 4))
        rgba[:, :3] = (0.17, 0.24, 0.31)          
        rgba[:, 3] = np.clip(alpha, 0, 1)
        return LineCollection(segs[:i], colors=rgba, linewidths=1.4, zorder=1)

    def update(i, show_title=True):
        ax_poly.clear()
        axes_cross(ax_poly)
        draw_polygon(ax_poly, frames[i], colors, diagonals=pairs, label=label)
        finish_axes(ax_poly, (-quad_window, quad_window),
                    (-quad_window, quad_window),
                    f"Cyclic folding of a {n}-gon, iteration {i}"
                    if show_title else None)

        ax_right.clear()
        if right_mode == "Two-diagonal trace":
            lc = fading_collection(i)
            if lc is not None:
                ax_right.add_collection(lc)
            ax_right.scatter(trace_x[i], trace_y[i], color=FOLD_LINE, s=42,
                             zorder=4, edgecolors="white", linewidths=0.8)
            ax_right.set_xlim(*xlim)
            ax_right.set_ylim(*ylim)
            ax_right.set_aspect("equal")
            ax_right.set_xlabel(rf"$|v_{{{pair_a[0]}}} - v_{{{pair_a[1]}}}|^2$")
            ax_right.set_ylabel(rf"$|v_{{{pair_b[0]}}} - v_{{{pair_b[1]}}}|^2$")
            if show_title:
                ax_right.set_title("Diagonal-length trace", pad=8)
        else:
            for p, (a, b) in enumerate(pairs):
                ax_right.plot(np.arange(i + 1), lengths[:i + 1, p],
                              color=line_colors[p], linewidth=1.2,
                              label=rf"$v_{{{a}}}v_{{{b}}}$")
                ax_right.scatter([i], [lengths[i, p]], color=line_colors[p],
                                 s=20, zorder=4, edgecolors="white",
                                 linewidths=0.6)
            ax_right.set_xlim(0, max(n_frames - 1, 1))
            
            ncol = 2 if len(pairs) <= 8 else 3
            rows = -(-len(pairs) // ncol)
            ax_right.set_ylim(0, lengths.max() * (1.1 + 0.08 * rows) + 1e-9)
            ax_right.set_xlabel("iteration $k$")
            ax_right.set_ylabel("squared diagonal length")
            ax_right.legend(fontsize=fs(8), ncol=ncol,
                            loc="upper right")
            if show_title:
                ax_right.set_title(f"{len(pairs)} diagonals tracked", pad=8)

    height_px = int(fig_h * ANIM_DPI) + 280
    return fig, update, n_frames, degenerate, height_px


def diagonal_dynamics_animation(v, steps, interval_ms, quad_window, op_name,
                                pair_a, pair_b, right_mode, pairs=None,
                                fade_steps=8, label=True):
    fig, update, n_frames, degenerate, height = _diagonal_dynamics_builder(
        v, steps, quad_window, op_name, pair_a, pair_b, right_mode, pairs,
        fade_steps, label, ANIM_DPI)
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval_ms, repeat=True)
    html = anim.to_jshtml()
    plt.close(fig)
    return html, degenerate, height


def diagonal_dynamics_still(v, steps, k, quad_window, op_name, pair_a, pair_b,
                            right_mode, pairs, show_title, fade_steps=8,
                            label=True):
    fig, update, n_frames, degenerate, _ = _diagonal_dynamics_builder(
        v, steps, quad_window, op_name, pair_a, pair_b, right_mode, pairs,
        fade_steps, label, STILL_DPI)
    k = min(k, n_frames - 1)
    update(k, show_title)
    return fig, degenerate, k






def show_animation(html_str, height_px):
    wrapped = f"""
    <div style="display:flex; justify-content:center; padding-top:16px;
                background:white;">
      {html_str}
    </div>
    """
    
    
    embed = getattr(getattr(st, "components", None), "v1", None)
    if embed is not None and hasattr(embed, "html"):
        embed.html(wrapped, height=height_px, scrolling=False)
    else:
        st.html(wrapped)


def warn_if_degenerate(degenerate):
    if degenerate:
        st.warning(
            "The polygon collapsed (two neighbouring vertices met), so the "
            "reflection is undefined and the run stopped early. Nudge a vertex "
            "and try again."
        )






def load_vertices(n):
    """Read the n vertex text inputs; returns None if anything is unparseable."""
    vals = []
    ok = True
    per_row = n if n <= 4 else (n + 1) // 2      
    for row_start in range(0, n, per_row):
        cols = st.columns(min(per_row, n - row_start))
        for offset, col in enumerate(cols):
            i = row_start + offset
            with col:
                x = st.text_input(f"v{i} x", key=f"vx_{i}")
                y = st.text_input(f"v{i} y", key=f"vy_{i}")
                try:
                    vals.append(complex(float(x), float(y)))
                except ValueError:
                    st.error(f"v{i} needs two numbers.")
                    ok = False
    return np.array(vals, dtype=complex) if ok else None


def seed_vertex_state(vs):
    for i, z in enumerate(vs):
        st.session_state[f"vx_{i}"] = f"{z.real:.3f}"
        st.session_state[f"vy_{i}"] = f"{z.imag:.3f}"


POLY_NAMES = {4: "Quadrilateral", 5: "Pentagon", 6: "Hexagon", 7: "Heptagon"}


def main():
    st.set_page_config(page_title="Polygon Dynamics", layout="wide")
    st.title("Cyclic Folding")
    st.caption("Reflect one vertex across the diagonal joining its neighbours, "
               "relabel, repeat.")

    st.subheader("1. Choose your polygon")
    n = st.radio(
        "Polygon",
        [4, 5, 6, 7],
        index=1,
        horizontal=True,
        format_func=lambda k: f"{k}-gon · {POLY_NAMES[k]}",
        label_visibility="collapsed",
        key="n_choice",
    )
    n_diag = len(diagonal_pairs(n))
    st.caption(f"Each step reflects v0 across the line through its neighbours "
               f"v1 and v{n - 1}. A {POLY_NAMES[n].lower()} has {n_diag} diagonals.")

    st.subheader("2. Place the vertices")
    if st.session_state.get("_n_cache") != n:
        seed_vertex_state(default_vertices(n))
        st.session_state["_n_cache"] = n

    b1, b2, _ = st.columns([1, 1, 4])
    with b1:
        if st.button("Reset to default", **WIDE_BUTTON):
            seed_vertex_state(default_vertices(n))
    with b2:
        if st.button("Randomize", **WIDE_BUTTON):
            seed = st.session_state.get("_seed", 0) + 1
            st.session_state["_seed"] = seed
            seed_vertex_state(random_vertices(n, np.random.default_rng(seed)))

    v = load_vertices(n)

    if v is not None:
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            fig = preview_figure(v)
            st.pyplot(fig, **WIDE_PLOT)
            download_row(render_bytes(fig), f"{n}gon_start", "preview")
            plt.close(fig)
            st.caption(f"Starting shape. The red segment joins v1 and v{n - 1} — "
                       f"that is the line v0 moves across on every step. Dotted "
                       f"lines are the short diagonals.")

    st.subheader("3. Pick a view and fold")
    mode = st.radio(
        "View",
        ["Plot Orbit", "Animate Folding", "Animate Folding (Centered)",
         "Visualize Diagonal Dynamics"],
        horizontal=True,
        key="mode_radio",
    )

    fold_type = st.selectbox("Fold Type", list(FOLD_FUNCS.keys()), index=0,
                             key="fold_type_select")
    st.caption(FOLD_NOTES[fold_type].format(last=n - 1))

    
    
    plotsize = 2.0
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        show_title = st.checkbox("Title on figure", value=False, key="fig_title")
    if mode != "Visualize Diagonal Dynamics":
        with c2:
            autofit = st.checkbox("Fit view to the data", value=True, key="autofit")
        with c3:
            if autofit:
                plotsize = None
                st.caption("The axes will be sized to whatever the orbit covers. "
                           "Uncheck to set a fixed window.")
            else:
                plotsize_input = st.text_input("Plot Size", value="2",
                                               key="plotsize_top")
                try:
                    plotsize = float(plotsize_input)
                    if plotsize <= 0:
                        raise ValueError
                except ValueError:
                    st.error("Plot size must be a positive number. Using 2.")
                    plotsize = 2.0

    ready = v is not None

    
    if mode == "Plot Orbit":
        c1, c2, c3 = st.columns(3)
        with c1:
            steps = st.slider("Iterations", 10, 50000, 2000, 10, key="orbit_iters")
        with c2:
            pointsize = st.slider("Point Size", 1, 10, 5, 1, key="orbit_pointsize")
        with c3:
            center = st.checkbox("Center each step", value=False, key="orbit_center")
        color_by_vertex = st.checkbox("Color points by vertex index", value=False,
                                      key="orbit_colors")

        if st.button("Generate orbit plot", type="primary",
                     **WIDE_BUTTON) and ready:
            with st.spinner("Rendering..."):
                fig, degenerate, clipped = plot_orbit(
                    v, steps, plotsize, pointsize, fold_type, color_by_vertex,
                    center, show_title)
                remember_figure("result_orbit", fig, degenerate=degenerate,
                                clipped=clipped, steps=steps)

        res = show_stored_figure("result_orbit", f"{n}gon_orbit_{fold_type}"
                                 .replace(" ", "_").lower())
        if res is not None:
            warn_if_degenerate(res["degenerate"])
            if res["clipped"] > 0.02:
                st.info(f"{res['clipped']:.0%} of the orbit falls outside this "
                        f"window. Tick 'Fit view to the data' or raise the plot "
                        f"size to see all of it — orbits of 6- and 7-gons often "
                        f"drift a long way from the origin.")

    
    elif mode in ("Animate Folding", "Animate Folding (Centered)"):
        center = mode.endswith("(Centered)")

        c1, c2 = st.columns(2)
        with c1:
            steps = st.slider("Animation Iterations", 1, 100, 20, 1, key="anim_iters")
        with c2:
            duration = st.slider("Frame Duration (ms)", 50, 1000, 200, 50,
                                 key="anim_duration")

        c1, c2, c3 = st.columns(3)
        with c1:
            show_orbit = st.checkbox("Show orbit background", value=False,
                                     key="anim_orbit")
        with c2:
            orbit_steps = (st.slider("Orbit Iterations", 100, 50000, 2000, 100,
                                     key="anim_orbit_iters") if show_orbit else 2000)
        with c3:
            orbit_alpha = (st.slider("Orbit Transparency", 0.0, 1.0, 0.3, 0.05,
                                     key="anim_alpha_orbit") if show_orbit else 0.3)

        c1, c2, c3 = st.columns(3)
        with c1:
            pointsize = (st.slider("Point Size", 1, 10, 2, 1, key="anim_pointsize")
                         if show_orbit else 2)
        with c2:
            label = st.checkbox("Label vertices", value=True, key="anim_label")
        with c3:
            diag = st.checkbox("Draw diagonals", value=False, key="anim_diag")

        if st.button("Generate animation", type="primary",
                     **WIDE_BUTTON, key="anim_button") and ready:
            with st.spinner("Rendering frames..."):
                html_anim, degenerate = animate_folding(
                    v, steps, duration, plotsize, pointsize, show_orbit,
                    orbit_steps, orbit_alpha, fold_type, center, label, diag,
                )
            st.session_state["result_anim"] = (html_anim, degenerate)

        if "result_anim" in st.session_state:
            html_anim, degenerate = st.session_state["result_anim"]
            warn_if_degenerate(degenerate)
            show_animation(html_anim, height_px=760)

        with st.expander("Still frame for print"):
            st.caption("Renders one iteration at print resolution with the "
                       "settings above, and offers it as PNG (300 dpi) or PDF.")
            k = st.number_input("Iteration", 0, steps, min(steps, 5), 1,
                                key="anim_still_k")
            if st.button("Render still frame", key="anim_still_btn",
                         **WIDE_BUTTON) and ready:
                fig, degenerate, k_used = folding_still(
                    v, steps, int(k), plotsize, pointsize, show_orbit,
                    orbit_steps, orbit_alpha, fold_type, center, label, diag,
                    show_title)
                remember_figure("result_anim_still", fig, degenerate=degenerate,
                                k=k_used)
            res = show_stored_figure("result_anim_still",
                                     f"{n}gon_iteration")
            if res is not None:
                warn_if_degenerate(res["degenerate"])
                st.caption(f"Iteration {res['k']}.")

    
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            steps = st.slider("Animation Iterations", 1, 250, 50, 1, key="dd_iters")
        with c2:
            duration = st.slider("Frame Duration (ms)", 50, 1000, 300, 50,
                                 key="dd_duration")
        with c3:
            quad_window = st.slider("Polygon Plot Size", 1.0, 3.0, 1.5, 0.25,
                                    key="dd_window")

        
        
        families = gap_families(n)
        gaps = families
        if len(families) > 1:
            span_names = {2: "Skip one (v0-v2)", 3: "Skip two (v0-v3)"}
            options = {"All": families}
            for g in families:
                options[span_names.get(g, f"Span {g}")] = [g]
            gaps = options[st.radio("Which diagonals", list(options),
                                    horizontal=True, key="dd_family")]

        pairs = diagonal_pairs(n, gaps)
        labels = [f"v{i}-v{j}" for i, j in pairs]

        right_mode = st.radio(
            "Right panel",
            ["Two-diagonal trace", "All diagonals vs iteration"],
            horizontal=True,
            key="dd_right_mode",
        )

        pair_a, pair_b = pairs[0], pairs[min(1, len(pairs) - 1)]
        if right_mode == "Two-diagonal trace":
            c1, c2 = st.columns(2)
            with c1:
                pair_a = pairs[labels.index(
                    st.selectbox("Horizontal axis", labels, index=0, key="dd_pa"))]
            with c2:
                pair_b = pairs[labels.index(
                    st.selectbox("Vertical axis", labels,
                                 index=min(1, len(labels) - 1), key="dd_pb"))]

        if steps > 100:
            st.caption(f"Every frame is embedded as an image, so {steps + 1} frames "
                       f"will take roughly {steps // 4} seconds to build.")

        if st.button("Generate diagonal dynamics", type="primary",
                     **WIDE_BUTTON, key="dd_button") and ready:
            with st.spinner("Rendering frames..."):
                html_anim, degenerate, height = diagonal_dynamics_animation(
                    v, steps, duration, quad_window, fold_type,
                    pair_a, pair_b, right_mode, pairs,
                )
            st.session_state["result_dd"] = (html_anim, degenerate, height)

        if "result_dd" in st.session_state:
            html_anim, degenerate, height = st.session_state["result_dd"]
            warn_if_degenerate(degenerate)
            show_animation(html_anim, height_px=height)

        with st.expander("Still frame for print"):
            st.caption("Renders both panels at one iteration at print "
                       "resolution, and offers them as PNG (300 dpi) or PDF.")
            k = st.number_input("Iteration", 0, steps, min(steps, 10), 1,
                                key="dd_still_k")
            if st.button("Render still frame", key="dd_still_btn",
                         **WIDE_BUTTON) and ready:
                fig, degenerate, k_used = diagonal_dynamics_still(
                    v, steps, int(k), quad_window, fold_type, pair_a, pair_b,
                    right_mode, pairs, show_title)
                remember_figure("result_dd_still", fig, degenerate=degenerate,
                                k=k_used)
            res = show_stored_figure("result_dd_still",
                                     f"{n}gon_diagonals", width_cols=(1, 6, 1))
            if res is not None:
                warn_if_degenerate(res["degenerate"])
                st.caption(f"Iteration {res['k']}.")


if __name__ == "__main__":
    if _under_streamlit():
        main()
    else:
        _relaunch_under_streamlit()