import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def create_spacetime_grid(size, scale):
    """Create a grid representing spacetime.
    
    Args:
        size (int): The range of the grid in each direction (-size to size).
        scale (int): The number of points in each direction.
    
    Returns:
        tuple: Meshgrid arrays for x, y, z coordinates.
    """
    x = np.linspace(-size, size, scale)
    y = np.linspace(-size, size, scale)
    z = np.linspace(-size, size, scale)
    x, y, z = np.meshgrid(x, y, z)
    return x, y, z

def refined_metric_tensor(x, y, z, t, bubble_radius, density, speed):
    """Refine the metric tensor for the warp bubble with additional terms.
    
    Args:
        x, y, z (ndarray): Meshgrid arrays for spatial coordinates.
        t (float): Current time step.
        bubble_radius (float): Radius of the warp bubble.
        density (float): Density of the warp bubble.
        speed (float): Speed of the warp bubble.
    
    Returns:
        tuple: Components of the metric tensor.
    """
    r = np.sqrt(x**2 + y**2 + z**2) - speed * t
    g_tt = -1
    g_xx = g_yy = g_zz = 1 + density * np.exp(-((r - bubble_radius) ** 2)) + 0.1 * density * np.exp(-((r - 2 * bubble_radius) ** 2))
    return g_tt, g_xx, g_yy, g_zz

def refined_energy_momentum_tensor(g_tt, g_xx, g_yy, g_zz):
    """Refine the energy-momentum tensor from the refined metric tensor.
    
    Args:
        g_tt, g_xx, g_yy, g_zz (ndarray): Components of the metric tensor.
    
    Returns:
        tuple: Components of the energy-momentum tensor.
    """
    T_tt = -g_tt
    T_xx = g_xx
    T_yy = g_yy
    T_zz = g_zz
    return T_tt, T_xx, T_yy, T_zz

def refined_warp_spacetime_dynamic(x, y, z, bubble_radius, density, t, speed):
    """Apply refined warp factor to spacetime grid to create a dynamic warp bubble using positive energy.
    
    Args:
        x, y, z (ndarray): Meshgrid arrays for spatial coordinates.
        bubble_radius (float): Radius of the warp bubble.
        density (float): Density of the warp bubble.
        t (float): Current time step.
        speed (float): Speed of the warp bubble.
    
    Returns:
        tuple: Warp effect and components of the energy-momentum tensor.
    """
    g_tt, g_xx, g_yy, g_zz = refined_metric_tensor(x, y, z, t, bubble_radius, density, speed)
    T_tt, T_xx, T_yy, T_zz = refined_energy_momentum_tensor(g_tt, g_xx, g_yy, g_zz)
    warp_effect = density * np.exp(-((np.sqrt(x**2 + y**2 + z**2) - bubble_radius - speed * t) ** 2))
    return warp_effect, T_tt, T_xx, T_yy, T_zz

def smooth_warp_effect(warp_effect, sigma=1):
    """Smooth the warp effect using a Gaussian filter.
    
    Args:
        warp_effect (ndarray): The warp effect applied to the grid.
        sigma (float): The standard deviation for Gaussian kernel.
    
    Returns:
        ndarray: The smoothed warp effect.
    """
    return gaussian_filter(warp_effect, sigma=sigma)

def plot_warped_spacetime_slices(x, y, z, warp_effect):
    """Plot slices of the warped spacetime grid in different planes.
    
    Args:
        x, y, z (ndarray): Meshgrid arrays for spatial coordinates.
        warp_effect (ndarray): The warp effect applied to the grid.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # XY plane
    slice_idx = warp_effect.shape[2] // 2
    warp_slice = warp_effect[:, :, slice_idx]
    axes[0].contourf(x[:, :, slice_idx], y[:, :, slice_idx], warp_slice, cmap='viridis')
    axes[0].set_title('XY Plane')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # YZ plane
    slice_idx = warp_effect.shape[0] // 2
    warp_slice = warp_effect[slice_idx, :, :]
    axes[1].contourf(y[slice_idx, :, :], z[slice_idx, :, :], warp_slice, cmap='viridis')
    axes[1].set_title('YZ Plane')
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('Z')
    
    # XZ plane
    slice_idx = warp_effect.shape[1] // 2
    warp_slice = warp_effect[:, slice_idx, :]
    axes[2].contourf(x[:, slice_idx, :], z[:, slice_idx, :], warp_slice, cmap='viridis')
    axes[2].set_title('XZ Plane')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Z')
    
    plt.colorbar(axes[0].contourf(x[:, :, slice_idx], y[:, :, slice_idx], warp_slice, cmap='viridis'), ax=axes, orientation='horizontal', fraction=0.05)
    plt.show()

def plot_energy_momentum(t, T_tt, T_xx, T_yy, T_zz):
    """Plot the energy-momentum tensor components over time.
    
    Args:
        t (ndarray): Array of time steps.
        T_tt, T_xx, T_yy, T_zz (ndarray): Energy-momentum tensor components over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(t, T_tt, label='T_tt')
    plt.plot(t, T_xx, label='T_xx')
    plt.plot(t, T_yy, label='T_yy')
    plt.plot(t, T_zz, label='T_zz')
    plt.xlabel('Time Step')
    plt.ylabel('Energy-Momentum Tensor Component')
    plt.title('Energy-Momentum Tensor Components Over Time')
    plt.legend()
    plt.show()

def refined_comprehensive_analysis(x, y, z, bubble_radius, density, speed, timesteps, time_interval):
    """Perform comprehensive analysis of the refined warp bubble dynamics.
    
    Args:
        x, y, z (ndarray): Meshgrid arrays for spatial coordinates.
        bubble_radius (float): Radius of the warp bubble.
        density (float): Density of the warp bubble.
        speed (float): Speed of the warp bubble.
        timesteps (int): Number of time steps for the simulation.
        time_interval (float): Time interval between each step.
    """
    t_values = np.arange(0, timesteps * time_interval, time_interval)
    T_tt_values = []
    T_xx_values = []
    T_yy_values = []
    T_zz_values = []

    for t in t_values:
        warp_effect, T_tt, T_xx, T_yy, T_zz = refined_warp_spacetime_dynamic(x, y, z, bubble_radius, density, t, speed)
        T_tt_values.append(np.mean(T_tt))
        T_xx_values.append(np.mean(T_xx))
        T_yy_values.append(np.mean(T_yy))
        T_zz_values.append(np.mean(T_zz))

        # Plot the warped spacetime slices at intervals
        if t % 1 == 0:  # Plot every timestep
            warp_effect_smoothed = smooth_warp_effect(warp_effect, sigma=2.0)
            plot_warped_spacetime_slices(x, y, z, warp_effect_smoothed)

    plot_energy_momentum(t_values, T_tt_values, T_xx_values, T_yy_values, T_zz_values)

def run_comprehensive_analysis(bubble_radius, density, speed, timesteps, time_interval):
    """Run the comprehensive analysis for a given set of parameters.
    
    Args:
        bubble_radius (float): Radius of the warp bubble.
        density (float): Density of the warp bubble.
        speed (float): Speed of the warp bubble.
        timesteps (int): Number of time steps for the simulation.
        time_interval (float): Time interval between each step.
    """
    x, y, z = create_spacetime_grid(new_grid_size, new_grid_scale)
    refined_comprehensive_analysis(x, y, z, bubble_radius, density, speed, timesteps, time_interval)

# Parameters for the grid and simulation
new_grid_size = 10
new_grid_scale = 100
timesteps = 20
time_interval = 0.1

# Configurations to test
configurations = [
    {"bubble_radius": 1.0, "density": 20.0, "speed": 1.0},
    {"bubble_radius": 3.0, "density": 10.0, "speed": 0.5},
    {"bubble_radius": 4.0, "density": 15.0, "speed": 2.0},
    {"bubble_radius": 5.0, "density": 5.0, "speed": 1.5},
]

# Run simulations for each configuration
for config in configurations:
    print(f"Running simulation with bubble_radius={config['bubble_radius']}, density={config['density']}, speed={config['speed']}")
    run_comprehensive_analysis(config['bubble_radius'], config['density'], config['speed'], timesteps, time_interval)
