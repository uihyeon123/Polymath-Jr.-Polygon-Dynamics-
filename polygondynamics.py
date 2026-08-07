#!/usr/bin/env python3
import importlib.util
import os
import subprocess
import sys


# =========================
# Green-button launcher
# =========================

def _under_streamlit() -> bool:
    """True when this script is being executed by the Streamlit runtime."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    try:
        # Asking outside a script run logs a noisy warning unless suppressed.
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
        pass  # harmless: worst case Streamlit asks and you press Enter


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

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


def _full_width(fn):
    """Streamlit renamed use_container_width to width; support both."""
    if "width" in inspect.signature(fn).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


WIDE_BUTTON = _full_width(st.button)
WIDE_PLOT = _full_width(st.pyplot)


# =========================
# Geometry helpers
# =========================

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


# =========================
# Folding operations (any n)
# =========================

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


# =========================
# Iteration
# =========================

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
    stack = np.array(frames)                 # (n_frames, n)
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
    return plt.cm.viridis(np.linspace(0.0, 0.85, n))


def draw_polygon(ax, v, colors, diagonals=None, label=False, s=30):
    x = np.append(v.real, v[0].real)
    y = np.append(v.imag, v[0].imag)
    ax.fill(x, y, color="lightgray", alpha=0.5)
    ax.plot(x, y, "k-", linewidth=1)

    for i, j in (diagonals or []):
        ax.plot([v[i].real, v[j].real], [v[i].imag, v[j].imag],
                "k:", linewidth=0.8, alpha=0.7)

    ax.scatter(v.real, v.imag, color=colors, s=s, zorder=5,
               edgecolors="black", linewidths=0.5)

    if label:
        for i, z in enumerate(v):
            ax.annotate(str(i), (z.real, z.imag), textcoords="offset points",
                        xytext=(6, 5), fontsize=8, color="black", zorder=6)


def preview_figure(v):
    """Small static picture of the starting polygon, with the fold line marked."""
    n = len(v)
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=90)
    # Only the short diagonals: a heptagon's 14 would bury the shape.
    draw_polygon(ax, v, vertex_colors(n), diagonals=diagonal_pairs(n, [2]),
                 label=True, s=26)
    # v1 to v[n-1] is the line (or segment) every operation acts on.
    ax.plot([v[1].real, v[-1].real], [v[1].imag, v[-1].imag],
            color="crimson", linewidth=1.7, alpha=0.85, zorder=4)
    lim = max(1.0, np.abs(v).max() * 1.25)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# =========================
# Orbit plot
# =========================

def plot_orbit(v, steps, plotsize, pointsize, op_name, color_by_vertex, center):
    """plotsize=None fits the axes to the data instead of using a fixed window."""
    frames, degenerate = iterate(v, steps, op_name, center=center)
    xs, ys = all_points(frames)
    n = xs.shape[1]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.subplots_adjust(top=0.92)

    if color_by_vertex:
        colors = vertex_colors(n)
        for i in range(n):
            ax.scatter(xs[:, i], ys[:, i], color=colors[i], s=pointsize,
                       alpha=0.6, label=f"v{i}")
        ax.legend(loc="upper right", fontsize=8, markerscale=2, framealpha=0.9)
    else:
        ax.scatter(xs.ravel(), ys.ravel(), color="black", s=pointsize, alpha=0.6)

    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    if plotsize is None:
        pad = 0.05 * max(np.ptp(xs), np.ptp(ys)) + 0.05
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)
        clipped = 0.0
    else:
        ax.set_xlim(-plotsize, plotsize)
        ax.set_ylim(-plotsize, plotsize)
        clipped = float(np.mean((np.abs(xs) > plotsize) | (np.abs(ys) > plotsize)))

    ax.set_aspect("equal")
    ax.set_title(f"Orbit of a {n}-gon over {len(frames) - 1} iterations", pad=12)
    return fig, degenerate, clipped


# =========================
# Folding animation
# =========================

def animate_folding(v, steps, interval_ms, plotsize, pointsize, show_orbit,
                    orbit_steps, orbit_alpha, op_name, center=False,
                    label=True, show_diagonals=False):
    n = len(v)
    colors = vertex_colors(n)
    diagonals = diagonal_pairs(n) if show_diagonals else None

    orbit_x = orbit_y = None
    if show_orbit:
        o_frames, _ = iterate(v, orbit_steps, op_name, center=center)
        ox, oy = all_points(o_frames)
        orbit_x, orbit_y = ox.ravel(), oy.ravel()

    frames, degenerate = iterate(v, steps, op_name, center=center)
    suffix = " (Centered)" if center else ""

    if plotsize is None:
        stack = np.array(frames)
        px = list(stack.real.ravel()) + (list(orbit_x) if show_orbit else [])
        py = list(stack.imag.ravel()) + (list(orbit_y) if show_orbit else [])
        pad = 0.05 * max(np.ptp(px), np.ptp(py)) + 0.05
        xlim = (min(px) - pad, max(px) + pad)
        ylim = (min(py) - pad, max(py) + pad)
    else:
        xlim = ylim = (-plotsize, plotsize)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
    fig.subplots_adjust(top=0.92)

    def update(i):
        ax.clear()
        if show_orbit:
            ax.scatter(orbit_x, orbit_y, color="gray", s=pointsize, alpha=orbit_alpha)
        draw_polygon(ax, frames[i], colors, diagonals, label, s=24)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(f"Iteration {i}{suffix}", pad=12)

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=interval_ms, repeat=True)
    plt.close(fig)
    return anim.to_jshtml(), degenerate


# =========================
# Diagonal dynamics
# =========================

def diagonal_lengths(frames, pairs):
    """Squared length of each diagonal at every iteration -> (n_frames, n_pairs)."""
    stack = np.array(frames)
    return np.array([np.abs(stack[:, i] - stack[:, j]) ** 2 for i, j in pairs]).T


def diagonal_dynamics_animation(v, steps, interval_ms, quad_window, op_name,
                                pair_a, pair_b, right_mode, pairs=None,
                                fade_steps=8, label=True):
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

    dpi, fig_h = 80, 5.5
    fig, (ax_poly, ax_right) = plt.subplots(1, 2, figsize=(11, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.10, wspace=0.28)

    def fading_collection(i):
        if i < 1:
            return None
        k = np.arange(i)
        alpha = np.full(i, 0.25)
        recent = k >= i - fade_steps
        alpha[recent] = 0.25 + 0.65 * (k[recent] - (i - fade_steps)) / max(fade_steps, 1)
        rgba = np.zeros((i, 4))
        rgba[:, :3] = 0.35
        rgba[:, 3] = np.clip(alpha, 0, 1)
        return LineCollection(segs[:i], colors=rgba, linewidths=1.4, zorder=1)

    def update(i):
        ax_poly.clear()
        draw_polygon(ax_poly, frames[i], colors, diagonals=pairs, label=label)
        ax_poly.set_xlim(-quad_window, quad_window)
        ax_poly.set_ylim(-quad_window, quad_window)
        ax_poly.set_aspect("equal")
        ax_poly.grid(True, alpha=0.3)
        ax_poly.set_title(f"Cyclic folding of a {n}-gon\nIteration {i}", fontsize=9)

        ax_right.clear()
        if right_mode == "Two-diagonal trace":
            lc = fading_collection(i)
            if lc is not None:
                ax_right.add_collection(lc)
            ax_right.scatter(trace_x[i], trace_y[i], color="black", s=40, zorder=4)
            ax_right.set_xlim(*xlim)
            ax_right.set_ylim(*ylim)
            ax_right.set_aspect("equal")
            ax_right.set_xlabel(f"|v{pair_a[0]} - v{pair_a[1]}|²")
            ax_right.set_ylabel(f"|v{pair_b[0]} - v{pair_b[1]}|²")
            ax_right.set_title("Diagonal-length trace", fontsize=9)
        else:
            cmap = plt.cm.plasma(np.linspace(0, 0.85, len(pairs)))
            for p, (a, b) in enumerate(pairs):
                ax_right.plot(np.arange(i + 1), lengths[:i + 1, p],
                              color=cmap[p], linewidth=1.2, label=f"v{a}-v{b}")
                ax_right.scatter([i], [lengths[i, p]], color=cmap[p], s=18, zorder=4)
            ax_right.set_xlim(0, max(n_frames - 1, 1))
            ax_right.set_ylim(0, lengths.max() * 1.1 + 1e-9)
            ax_right.set_xlabel("iteration")
            ax_right.set_ylabel("squared length")
            ax_right.legend(fontsize=6.5, ncol=2 if len(pairs) <= 8 else 3,
                            loc="upper right", framealpha=0.9)
            ax_right.set_title(f"{len(pairs)} diagonals tracked", fontsize=9)
        ax_right.grid(True, alpha=0.3)

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval_ms, repeat=True)
    plt.close(fig)
    return anim.to_jshtml(), degenerate, int(fig_h * dpi) + 280


# =========================
# Shared helper
# =========================

def show_animation(html_str, height_px):
    wrapped = f"""
    <div style="display:flex; justify-content:center; padding-top:16px;">
      {html_str}
    </div>
    """
    # The animation carries its own JavaScript player, so it needs a real
    # component iframe. st.html strips scripts, so it is only a last resort.
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


# =========================
# Streamlit UI
# =========================

def load_vertices(n):
    """Read the n vertex text inputs; returns None if anything is unparseable."""
    vals = []
    ok = True
    per_row = n if n <= 4 else (n + 1) // 2      # 4, then 3+2, 3+3, 4+3
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
    if mode != "Visualize Diagonal Dynamics":
        c1, c2 = st.columns([1, 2])
        with c1:
            autofit = st.checkbox("Fit view to the data", value=True, key="autofit")
        with c2:
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

    # ---------- Plot Orbit ----------
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
            fig, degenerate, clipped = plot_orbit(v, steps, plotsize, pointsize,
                                                  fold_type, color_by_vertex, center)
            warn_if_degenerate(degenerate)
            if clipped > 0.02:
                st.info(f"{clipped:.0%} of the orbit falls outside this window. "
                        f"Tick 'Fit view to the data' or raise the plot size to "
                        f"see all of it — orbits of 6- and 7-gons often drift a "
                        f"long way from the origin.")
            _, mid, _ = st.columns([1, 2.5, 1])
            with mid:
                st.pyplot(fig, **WIDE_PLOT)
            plt.close(fig)

    # ---------- Animations ----------
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
            warn_if_degenerate(degenerate)
            show_animation(html_anim, height_px=720)

    # ---------- Diagonal dynamics ----------
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

        # Hexagons and heptagons have two families of diagonals; tracking one
        # family at a time keeps both panels readable.
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
            warn_if_degenerate(degenerate)
            show_animation(html_anim, height_px=height)


if __name__ == "__main__":
    if _under_streamlit():
        main()
    else:
        _relaunch_under_streamlit()