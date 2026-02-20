import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Grid and Metric Setup
# -----------------------------
def create_spacetime_grid(size, scale):
    x = np.linspace(-size, size, scale)
    y = np.linspace(-size, size, scale)
    z = np.linspace(-size, size, scale)
    return np.meshgrid(x, y, z)

def refined_metric_tensor(x, y, z, t, bubble_radius, density, speed, k, omega, sigma):
    r = np.sqrt(x**2 + y**2 + z**2) - speed * t
    wave = 1 + np.cos(k * x + omega * t)
    g_tt = -1
    g_xx = g_yy = g_zz = 1 + density * np.exp(-((r - bubble_radius) ** 2) / sigma**2) * wave
    return g_tt, g_xx, g_yy, g_zz

def refined_energy_momentum_tensor(g_tt, g_xx, g_yy, g_zz):
    return -g_tt, g_xx, g_yy, g_zz

def refined_warp_spacetime_dynamic(x, y, z, bubble_radius, density, t, speed, k, omega, sigma):
    g_tt, g_xx, g_yy, g_zz = refined_metric_tensor(x, y, z, t, bubble_radius, density, speed, k, omega, sigma)
    T_tt, T_xx, T_yy, T_zz = refined_energy_momentum_tensor(g_tt, g_xx, g_yy, g_zz)
    r = np.sqrt(x**2 + y**2 + z**2) - speed * t
    warp_effect = density * np.exp(-((r - bubble_radius) ** 2) / sigma**2) * (1 + np.cos(k * x + omega * t))
    return warp_effect, T_tt, T_xx, T_yy, T_zz

def smooth_warp_effect(warp_effect, sigma=1):
    return gaussian_filter(warp_effect, sigma=sigma)

# -----------------------------
# Live Animation
# -----------------------------
def run_live_simulation(config, grid_size, grid_scale, timesteps, time_interval, k, omega, sigma):
    x, y, z = create_spacetime_grid(grid_size, grid_scale)

    t_values = np.arange(0, timesteps * time_interval, time_interval)
    T_tt_values, T_xx_values, T_yy_values, T_zz_values = [], [], [], []

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.tight_layout()

    # Initialize plots
    slice_idx_xy = grid_scale // 2
    slice_idx_yz = grid_scale // 2
    slice_idx_xz = grid_scale // 2

    XY = axes[0,0].imshow(np.zeros((grid_scale, grid_scale)), origin='lower', cmap='viridis')
    YZ = axes[0,1].imshow(np.zeros((grid_scale, grid_scale)), origin='lower', cmap='viridis')
    XZ = axes[0,2].imshow(np.zeros((grid_scale, grid_scale)), origin='lower', cmap='viridis')

    EM_plot, = axes[1,0].plot([], [], label='T_tt')
    axes[1,0].set_xlim(0, timesteps)
    axes[1,0].set_ylim(-1, config["density"]*5)
    axes[1,0].set_title('T_tt over Time')
    axes[1,0].legend()

    def update(frame):
        t = t_values[frame]
        warp, T_tt, T_xx, T_yy, T_zz = refined_warp_spacetime_dynamic(
            x, y, z,
            bubble_radius=config["bubble_radius"],
            density=config["density"],
            t=t,
            speed=config["speed"],
            k=k, omega=omega, sigma=sigma
        )
        warp_smooth = smooth_warp_effect(warp, sigma=2.0)

        XY.set_data(warp_smooth[:,:,slice_idx_xy])
        YZ.set_data(warp_smooth[slice_idx_yz,:,:])
        XZ.set_data(warp_smooth[:,slice_idx_xz,:])

        T_tt_values.append(np.mean(T_tt))
        EM_plot.set_data(np.arange(len(T_tt_values)), T_tt_values)

        return XY, YZ, XZ, EM_plot

    anim = FuncAnimation(fig, update, frames=timesteps, interval=200, blit=False, repeat=False)
    plt.show()

# -----------------------------
# Simulation Parameters
# -----------------------------
new_grid_size = 10
new_grid_scale = 50  # reduced grid for faster animation
timesteps = 20
time_interval = 0.1
k = 1.0
omega = 2.0
sigma = 2.0

configurations = [
    {"bubble_radius": 1.0, "density": 20.0, "speed": 1.0},
    {"bubble_radius": 3.0, "density": 10.0, "speed": 0.5},
]

for cfg in configurations:
    print(f"Running live simulation: bubble_radius={cfg['bubble_radius']}, density={cfg['density']}, speed={cfg['speed']}")
    run_live_simulation(cfg, new_grid_size, new_grid_scale, timesteps, time_interval, k, omega, sigma)