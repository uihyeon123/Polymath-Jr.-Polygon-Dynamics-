import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import integrate

st.set_page_config(page_title="Polygon Dynamics", layout="wide")

st.title("Cyclic Folding")


# =========================
# Folding Functions
# =========================

def fold(v0, v1, v2, v3): #This operation is responsible for folding operation
    v0_folded = (np.conj(v0 - v1) * (v3 - v1)) / np.conj(v3 - v1) + v1
    new_v0 = v1
    new_v1 = v2
    new_v2 = v3
    new_v3 = v0_folded
    return new_v0, new_v1, new_v2, new_v3

def fold_centered(v0, v1, v2, v3):
    v0, v1, v2, v3 = fold(v0, v1, v2, v3)
    center = (v0 + v1 + v2 + v3) / 4
    return v0 - center, v1 - center, v2 - center, v3 - center


def fold_reverse(v0, v1, v2, v3): #This operation is folding operation, 
    v0_folded = (np.conj(v0 - v1) * (v3 - v1)) / np.conj(v3 - v1) + v1
    new_v0 = v3
    new_v1 = v0_folded
    new_v2 = v1
    new_v3 = v2
    return new_v0, new_v1, new_v2, new_v3


def fold_reverse_centered(v0, v1, v2, v3):
    v0, v1, v2, v3 = fold_reverse(v0, v1, v2, v3)
    center = (v0 + v1 + v2 + v3) / 4
    return v0 - center, v1 - center, v2 - center, v3 - center


def recut(v0, v1, v2, v3): #This function is responsible for recutting operation
    theta1 = np.angle(v2 - v1)
    theta2 = np.angle((v3 - v1) / (v2 - v1))
    theta3 = np.angle((v0 - v1) / (v3 - v1))
    translation = (np.linalg.norm(v3 - v1))/2 * 1j

    v0_folded = (np.conj((v0 - v1) * np.exp(1j*(np.pi/2 - theta1 - theta2)) - translation) + translation) * np.exp(-1j*(np.pi/2 - theta1 - theta2)) + v1
    new_v0 = v1
    new_v1 = v2
    new_v2 = v3
    new_v3 = v0_folded
    return new_v0, new_v1, new_v2, new_v3


def recut_centered(v0, v1, v2, v3):
    v0, v1, v2, v3 = recut(v0, v1, v2, v3)
    center = (v0 + v1 + v2 + v3) / 4
    return v0 - center, v1 - center, v2 - center, v3 - center

def snap(v0, v1, v2, v3):
    v0_snapped = (v1 + v3)/2
    new_v0 = v1
    new_v1 = v2
    new_v2 = v3
    new_v3 = v0_snapped
    return new_v0, new_v1, new_v2, new_v3

def snap_centered(v0, v1, v2, v3):
    v0, v1, v2, v3 = snap(v0, v1, v2, v3)
    center = (v0 + v1 + v2 + v3) / 4
    return v0 - center, v1 - center, v2 - center, v3 - center

def fold_then_recut(v0, v1, v2, v3):
    v0_folded = (np.conj(v0 - v1) * (v3 - v1)) / np.conj(v3 - v1) + v1
    new_v0 = v1
    new_v1 = v2
    new_v2 = v3
    new_v3 = v0_folded

    theta1 = np.angle(new_v2 - new_v1)
    theta2 = np.angle((new_v3 - new_v1) / (new_v2 - new_v1))
    theta3 = np.angle((new_v0 - new_v1) / (new_v3 - new_v1))
    translation = (np.linalg.norm(new_v3 - new_v1))/2 * 1j

    v0_folded = (np.conj((new_v0 - new_v1) * np.exp(1j*(np.pi/2 - theta1 - theta2)) - translation) + translation) * np.exp(-1j*(np.pi/2 - theta1 - theta2)) + new_v1
    newnew_v0 = new_v1
    newnew_v1 = new_v2
    newnew_v2 = new_v3
    newnew_v3 = v0_folded
    return newnew_v0, newnew_v1, newnew_v2, newnew_v3
    

def fold_then_recut_centered(v0, v1, v2, v3):
    v0, v1, v2, v3 = fold_then_recut(v0, v1, v2, v3)
    center = (v0 + v1 + v2 + v3) / 4
    return v0 - center, v1 - center, v2 - center, v3 - center




FOLD_FUNCS = {
    "Diagonal Reflection": (fold, fold_centered),
    "Diagonal Reflection in Reverse Direction": (fold_reverse, fold_reverse_centered),
    "Perpendicular Bisector Reflection": (recut, recut_centered),
    "Snap": (snap, snap_centered),
    "Fold then Recut": (fold_then_recut, fold_then_recut_centered)
    
}


# =========================
# Orbit Plot
# =========================

def plot_orbit_to_image(mu, nu, iters, plotsize, pointsize=5, fold_type="Diagonal Reflection"):
    fold_fn, _ = FOLD_FUNCS[fold_type]

    v0 = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
    v1 = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
    v2 = v1.conjugate()
    v3 = v0.conjugate()

    all_points = [v0, v1, v2, v3]

    for _ in range(iters):
        v0, v1, v2, v3 = fold_fn(v0, v1, v2, v3)
        all_points.extend([v0, v1, v2, v3])

    x = [z.real for z in all_points]
    y = [z.imag for z in all_points]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.subplots_adjust(top=0.92)

    ax.scatter(x, y, color="black", s=pointsize, alpha=0.6)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(-plotsize, plotsize)
    ax.set_ylim(-plotsize, plotsize)

    ax.set_title(f"Orbit over {iters} iterations (μ={mu}, ν={nu})", pad=12)

    return fig


# =========================
# Animation (Standard)
# =========================

def animate_folding(
    mu, nu, iters, duration, plotsize=3,
    pointsize=2, orbit=False,
    iters_orbit=1000, alpha_orbit=0.3,
    fold_type="Diagonal Reflection",
):
    fold_fn, _ = FOLD_FUNCS[fold_type]

    if orbit:
        v0_o = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
        v1_o = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
        v2_o = v1_o.conjugate()
        v3_o = v0_o.conjugate()

        orbit_points = [v0_o, v1_o, v2_o, v3_o]

        for _ in range(iters_orbit):
            v0_o, v1_o, v2_o, v3_o = fold_fn(v0_o, v1_o, v2_o, v3_o)
            orbit_points.extend([v0_o, v1_o, v2_o, v3_o])

        orbit_x = [z.real for z in orbit_points]
        orbit_y = [z.imag for z in orbit_points]

    v0 = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
    v1 = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
    v2 = v1.conjugate()
    v3 = v0.conjugate()

    frames = [(v0, v1, v2, v3)]

    for _ in range(iters):
        v0, v1, v2, v3 = fold_fn(v0, v1, v2, v3)
        frames.append((v0, v1, v2, v3))

    fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
    fig.subplots_adjust(top=0.92)

    def update(frame_num):
        ax.clear()

        if orbit:
            ax.scatter(orbit_x, orbit_y, color="gray", s=pointsize, alpha=alpha_orbit)

        v0, v1, v2, v3 = frames[frame_num]
        vertices = [v0, v1, v2, v3]

        x = [z.real for z in vertices] + [v0.real]
        y = [z.imag for z in vertices] + [v0.imag]

        ax.fill(x, y, color="lightgray", alpha=0.5)
        ax.plot(x, y, "k-", linewidth=1)
        ax.scatter([z.real for z in vertices], [z.imag for z in vertices],
                   color="black", s=20, zorder=5)

        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-plotsize, plotsize)
        ax.set_ylim(-plotsize, plotsize)
        ax.set_title(f"Iteration {frame_num}", pad=12)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=duration, repeat=True)
    plt.close()
    return anim.to_jshtml()


# =========================
# Animation (Centered)
# =========================

def animate_folding_centered(
    mu, nu, iters, duration, plotsize=2.0,
    pointsize=2, orbit=False,
    iters_orbit=1000, alpha_orbit=0.3,
    fold_type="Diagonal Reflection",
):
    _, fold_c_fn = FOLD_FUNCS[fold_type]

    if orbit:
        v0_o = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
        v1_o = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
        v2_o = v1_o.conjugate()
        v3_o = v0_o.conjugate()

        center = (v0_o + v1_o + v2_o + v3_o) / 4
        v0_o -= center
        v1_o -= center
        v2_o -= center
        v3_o -= center

        orbit_points = [v0_o, v1_o, v2_o, v3_o]

        for _ in range(iters_orbit):
            v0_o, v1_o, v2_o, v3_o = fold_c_fn(v0_o, v1_o, v2_o, v3_o)
            orbit_points.extend([v0_o, v1_o, v2_o, v3_o])

        orbit_x = [z.real for z in orbit_points]
        orbit_y = [z.imag for z in orbit_points]

    v0 = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
    v1 = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
    v2 = v1.conjugate()
    v3 = v0.conjugate()

    center = (v0 + v1 + v2 + v3) / 4
    v0 -= center
    v1 -= center
    v2 -= center
    v3 -= center

    frames = [(v0, v1, v2, v3)]

    for _ in range(iters):
        v0, v1, v2, v3 = fold_c_fn(v0, v1, v2, v3)
        frames.append((v0, v1, v2, v3))

    fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
    fig.subplots_adjust(top=0.92)

    def update(frame_num):
        ax.clear()

        if orbit:
            ax.scatter(orbit_x, orbit_y, color="gray", s=pointsize, alpha=alpha_orbit)

        v0, v1, v2, v3 = frames[frame_num]
        vertices = [v0, v1, v2, v3]

        x = [z.real for z in vertices] + [v0.real]
        y = [z.imag for z in vertices] + [v0.imag]

        ax.fill(x, y, color="lightgray", alpha=0.5)
        ax.plot(x, y, "k-", linewidth=1)
        ax.scatter([z.real for z in vertices], [z.imag for z in vertices],
                   color="black", s=20, zorder=5)

        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-plotsize, plotsize)
        ax.set_ylim(-plotsize, plotsize)
        ax.set_title(f"Iteration {frame_num} (Centered)", pad=12)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=duration, repeat=True)
    plt.close()
    return anim.to_jshtml()


# =========================
# Diagonal Dynamics Helpers
# =========================

def _dd_initial_vertices(mu, nu):
    v0 = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
    v1 =  np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
    v2 = v1.conjugate()
    v3 = v0.conjugate()
    center = (v0 + v1 + v2 + v3) / 4
    return v0 - center, v1 - center, v2 - center, v3 - center


def _dd_diagonal_pair(v0, v1, v2, v3):
    x = abs(v0 - v2)**2
    y = abs(v1 - v3)**2
    return x, y


def _dd_QLC(mu, nu):
    s = 2 - mu - nu
    Q = 4*(mu**2 + nu**2) + 2*s**2
    L = (4*mu**2 - s**2) * (4*nu**2 - s**2)
    C = (16*mu**2*nu**2 - s**4) * (4*(mu**2 + nu**2) - 2*s**2)
    return Q, L, C


def _dd_g2g3(mu, nu):
    Q, L, C = _dd_QLC(mu, nu)
    g2 = (1/12)*Q**4 + (4/3)*Q**2*L + (4/3)*L**2 - 2*Q*C
    g3 = ((-1/216)*Q**6 - (1/9)*Q**4*L + (1/6)*Q**3*C
          - (5/9)*L**2*Q**2 + (8/27)*L**3 + (4/3)*C*L*Q - C**2)
    return g2, g3


def _dd_Hx(mu, nu):
    Q, L, C = _dd_QLC(mu, nu)
    return (Q**2 + 8*L) / 12


def _dd_integrand(t, g2, g3):
    val = 4*t**3 - g2*t - g3
    return 1.0 / np.sqrt(val) if val > 0 else 0.0


def _dd_compute_rho(mu, nu):
    Q, L, C = _dd_QLC(mu, nu)
    g2, g3  = _dd_g2g3(mu, nu)
    hx      = _dd_Hx(mu, nu)

    roots = np.roots([4, 0, -g2, -g3])
    real_roots = roots[np.abs(roots.imag) < 1e-6].real
    if len(real_roots) == 0:
        return np.nan
    e2 = np.max(real_roots)

    num, _ = integrate.quad(_dd_integrand, hx, np.inf, args=(g2, g3))
    den, _ = integrate.quad(_dd_integrand, e2, np.inf, args=(g2, g3))

    raw = num / (2 * den)
    return 1.0 - raw if C < 0 else raw


def _dd_is_degenerate(mu, nu, tol=1e-9):
    return abs(mu + nu - 1) < tol or abs(mu - nu) < tol


def diagonal_dynamics_animation(mu, nu, iters, duration_ms, quad_window=1.5, resolution=800, fade_steps=8, fold_type="Diagonal Reflection"):
    _, fold_c_fn = FOLD_FUNCS[fold_type]
    is_diagonal = (fold_type == "Diagonal Reflection")

    degenerate = _dd_is_degenerate(mu, nu) if is_diagonal else False

    if is_diagonal and not degenerate:
        rho = _dd_compute_rho(mu, nu)
        theta_step = 2 * np.pi * rho

    v0, v1, v2, v3 = _dd_initial_vertices(mu, nu)
    frames = [(v0, v1, v2, v3)]
    orbit_x, orbit_y = [], []

    for i in range(iters + 1):
        if i > 0:
            v0, v1, v2, v3 = fold_c_fn(v0, v1, v2, v3)
            frames.append((v0, v1, v2, v3))

        x_diag, y_diag = _dd_diagonal_pair(v0, v1, v2, v3)
        if is_diagonal and i % 2 == 1:
            x_diag, y_diag = y_diag, x_diag

        orbit_x.append(x_diag)
        orbit_y.append(y_diag)

    orbit_x = np.array(orbit_x)
    orbit_y = np.array(orbit_y)

    if is_diagonal:
        curve_pad = 1.0
        Q, L, C = _dd_QLC(mu, nu)

        def F(x, y):
            return x**2*y + x*y**2 - Q*x*y - L*(x+y) + C

        xmin_raw = orbit_x.min() - curve_pad
        xmax_raw = orbit_x.max() + curve_pad
        ymin_raw = orbit_y.min() - curve_pad
        ymax_raw = orbit_y.max() + curve_pad

        x_width = xmax_raw - xmin_raw
        y_width = ymax_raw - ymin_raw

        if x_width < y_width:
            diff = (y_width - x_width) / 2
            xmin, xmax = xmin_raw - diff, xmax_raw + diff
            ymin, ymax = ymin_raw, ymax_raw
        else:
            diff = (x_width - y_width) / 2
            xmin, xmax = xmin_raw, xmax_raw
            ymin, ymax = ymin_raw - diff, ymax_raw + diff

        x_vals = np.linspace(xmin, xmax, resolution)
        y_vals = np.linspace(ymin, ymax, resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = F(X, Y)
    else:
        curve_pad = 1.0
        xmin, xmax = orbit_x.min() - curve_pad, orbit_x.max() + curve_pad
        ymin, ymax = orbit_y.min() - curve_pad, orbit_y.max() + curve_pad

    if is_diagonal and not degenerate:
        circle_pts_list = []
        theta = 0
        for i in range(iters + 1):
            circle_pts_list.append((np.cos(theta), np.sin(theta)))
            if i % 2 == 1:
                theta += theta_step
        circle_pts = np.array(circle_pts_list)  # shape: (iters+1, 2)

    show_circle_panel = is_diagonal and not degenerate
    n_panels = 3 if show_circle_panel else 2

    dpi = 80
    fig_w = 5.5 * n_panels
    fig_h = 5.5
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.08, wspace=0.35)

    ax_quad  = axes[0]
    ax_curve = axes[1]
    ax_circle = axes[2] if show_circle_panel else None

    def draw_fading_path(ax, xs, ys, i):
        for k in range(i):
            alpha = (0.5 + 0.5*(k - (i - fade_steps)) / fade_steps
                     if k >= i - fade_steps else 0.5)
            ax.plot(xs[k:k+2], ys[k:k+2],
                    color="gray", linewidth=1.4, alpha=alpha,
                    solid_capstyle='round', solid_joinstyle='round', zorder=1)

    def update(i):
        ax_quad.clear()
        v0, v1, v2, v3 = frames[i]
        verts = [v0, v1, v2, v3]
        xq = [z.real for z in verts] + [v0.real]
        yq = [z.imag for z in verts] + [v0.imag]

        ax_quad.fill(xq, yq, color="lightgray", alpha=0.5)
        ax_quad.plot(xq, yq, "k-", linewidth=1)
        ax_quad.plot([v0.real, v2.real], [v0.imag, v2.imag], "k:", linewidth=1)
        ax_quad.plot([v1.real, v3.real], [v1.imag, v3.imag], "k:", linewidth=1)
        ax_quad.scatter([z.real for z in verts], [z.imag for z in verts],
                        color="black", s=30, zorder=5)
        ax_quad.set_xlim(-quad_window, quad_window)
        ax_quad.set_ylim(-quad_window, quad_window)
        ax_quad.set_aspect("equal")
        ax_quad.grid(True, alpha=0.3)
        ax_quad.set_title(f"Cyclic Folding\nIteration {i}", fontsize=9)

        ax_curve.clear()
        draw_fading_path(ax_curve, orbit_x, orbit_y, i)
        if is_diagonal:
            ax_curve.contour(X, Y, Z, levels=[0], colors="black", linewidths=1.5, zorder=3)
        ax_curve.scatter(orbit_x[i], orbit_y[i], color="black", s=40, zorder=4)
        ax_curve.set_xlim(xmin, xmax)
        ax_curve.set_ylim(ymin, ymax)
        ax_curve.set_aspect("equal")
        ax_curve.grid(True, alpha=0.3)
        title = "Dynamics of π on Σ" if is_diagonal else "Diagonal-length trace"
        ax_curve.set_title(title, fontsize=9)

        if show_circle_panel:
            ax_circle.clear()
            if i > 0:
                draw_fading_path(ax_circle, circle_pts[:, 0], circle_pts[:, 1], i)
            circle_patch = plt.Circle((0, 0), 1, fill=False, color="black",
                                      linewidth=1.5, zorder=3)
            ax_circle.add_patch(circle_patch)
            ax_circle.scatter(circle_pts[i, 0], circle_pts[i, 1],
                               color="black", s=40, zorder=4)
            ax_circle.set_xlim(-1.2, 1.2)
            ax_circle.set_ylim(-1.2, 1.2)
            ax_circle.set_aspect("equal")
            ax_circle.grid(True, alpha=0.3)
            ax_circle.set_title(f"Rotation on Circle csp. τ\nρ ≈ {rho:.5f}", fontsize=9)

    anim = FuncAnimation(fig, update, frames=iters + 1,
                         interval=duration_ms, repeat=True)
    plt.close()

    render_height = int(fig_h * dpi) + 280
    return anim.to_jshtml(), degenerate, render_height


# =========================
# Shared helper: centered iframe
# =========================

def show_animation(html_str, height_px):
    wrapped = f"""
    <div style="display:flex; justify-content:center; padding-top:16px;">
      {html_str}
    </div>
    """
    st.components.v1.html(wrapped, height=height_px, scrolling=False)


# =========================
# Streamlit UI
# =========================

mode = st.radio(
    "",
    ["Plot Orbit", "Animate Folding", "Animate Folding (Centered)", "Visualize Diagonal Dynamics"],
    horizontal=True,
    label_visibility="collapsed",
    key="mode_radio",
)

col1, col2, col3 = st.columns(3)

with col1:
    mu_input = st.text_input("μ (mu)", value="0.3", key="mu_top")
    try:
        mu = float(mu_input)
        if not (0.0 <= mu <= 1.0):
            st.error("μ must be between 0 and 1.")
            mu = None
    except ValueError:
        st.error("μ must be a valid number.")
        mu = None

with col2:
    nu_input = st.text_input("ν (nu)", value="0.4", key="nu_top")
    try:
        nu = float(nu_input)
        if not (0.0 <= nu <= 1.0):
            st.error("ν must be between 0 and 1.")
            nu = None
    except ValueError:
        st.error("ν must be a valid number.")
        nu = None

with col3:
    if mode != "Visualize Diagonal Dynamics":
        plotsize_input = st.text_input("Plot Size", value="2", key="plotsize_top")
        try:
            plotsize = float(plotsize_input)
            if plotsize <= 0:
                st.error("Plot size must be positive.")
                plotsize = 2.0
        except ValueError:
            st.error("Plot size must be a number.")
            plotsize = 2.0
    else:
        plotsize = 3

# Fold-type selector — applies to all modes except Diagonal Dynamics,
# whose invariant-curve/rotation-number machinery is derived specifically
# for the diagonal-reflection fold.
if mode != "Visualize Diagonal Dynamics":
    fold_type = st.selectbox(
        "Fold Type",
        list(FOLD_FUNCS.keys()),
        index=0,
        key="fold_type_select",
    )
else:
    fold_type = st.selectbox(
        "Fold Type",
        list(FOLD_FUNCS.keys()),
        index=0,
        key="fold_type_select",
    )
    if fold_type != "Diagonal Reflection":
        st.caption(
            ""
        )


# =========================
# Plot Orbit Mode
# =========================

if mode == "Plot Orbit":

    col1, col2 = st.columns(2)

    with col1:
        iters = st.slider("Iterations", 10, 5000, 2000, 10, key="orbit_iters")

    with col2:
        pointsize = st.slider("Point Size", 1, 10, 5, 1, key="orbit_pointsize")

    if st.button("Generate Orbit Plot", type="primary", use_container_width=True) and mu is not None and nu is not None:
        fig = plot_orbit_to_image(mu, nu, iters, plotsize, pointsize, fold_type)
        buf_col1, buf_col2, buf_col3 = st.columns([1, 2.5, 1])
        with buf_col2:
            st.pyplot(fig, use_container_width=True)
        plt.close()


# =========================
# Animation Modes
# =========================

elif mode in ("Animate Folding", "Animate Folding (Centered)"):

    col1, col2 = st.columns(2)

    with col1:
        iters = st.slider("Animation Iterations", 1, 100, 20, 1, key="anim_iters")

    with col2:
        duration = st.slider("Frame Duration (ms)", 50, 1000, 200, 50, key="anim_duration")

    col1, col2, col3 = st.columns(3)

    with col1:
        orbit = st.checkbox("Show Orbit Background", value=False, key="anim_orbit")

    with col2:
        iters_orbit = (
            st.slider("Orbit Iterations", 100, 5000, 2000, 100, key="anim_orbit_iters")
            if orbit else 2000
        )

    with col3:
        alpha_orbit = (
            st.slider("Orbit Transparency", 0.0, 1.0, 0.3, 0.05, key="anim_alpha_orbit")
            if orbit else 0.3
        )

    if orbit:
        pointsize = st.slider("Point Size", 1, 10, 2, 1, key="anim_pointsize")
    else:
        pointsize = 2

    if mode == "Animate Folding":
        label = "Generate Animation"
        func = animate_folding
    else:
        label = "Generate Centered Animation"
        func = animate_folding_centered

    if st.button(label, type="primary", use_container_width=True, key="anim_button") and mu is not None and nu is not None:
        html_anim = func(
            mu, nu, iters, duration,
            plotsize, pointsize,
            orbit, iters_orbit, alpha_orbit,
            fold_type,
        )
        show_animation(html_anim, height_px=720)


# =========================
# Visualize Diagonal Dynamics
# =========================

else:

    col1, col2, col3 = st.columns(3)

    with col1:
        iters = st.slider("Animation Iterations", 1, 500, 50, 1, key="dd_iters")

    with col2:
        duration = st.slider("Frame Duration (ms)", 50, 1000, 300, 50, key="dd_duration")

    with col3:
        quad_window = st.slider("Quadrilateral Plot Size", 1.0, 3.0, 1.5, 0.25, key="dd_quad_window")

    if (
        fold_type == "Diagonal Reflection"
        and mu is not None and nu is not None
        and _dd_is_degenerate(mu, nu)
    ):
        st.warning(
            "Degenerate case detected (μ + ν = 1 or μ = ν). "
            "The rotation number is undefined and the circle panel will be hidden."
        )

    if st.button("Generate Diagonal Dynamics", type="primary", use_container_width=True, key="dd_button") and mu is not None and nu is not None:
        html_anim, degen, render_height = diagonal_dynamics_animation(
            mu, nu, iters, duration, quad_window, fold_type=fold_type
        )
        show_animation(html_anim, height_px=render_height)
