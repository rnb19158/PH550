
#!/usr/bin/env python3
"""
Created on Fri May 30 21:09:03 2025

@author: zhanmingmei

Kerr Cavity Simulator with Large Domain but "Zoomed-Out" k-Space Plot

Split-step 2D Lugiato-Lefever (fixed dt). We use a log-scale power spectrum
and artificially expand the plotting extent in k-space so we can "zoom out."
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

###############################################################################
# Utility & Simulation Functions
###############################################################################

def generate_noise(Nx, Ny, amp, noise_amplitude, dtype=np.complex64):
    """
    Creates a field of random complex noise of amplitude ~ amp*noise_amplitude.
    """
    noise_re = np.random.uniform(-1, 1, (Ny, Nx)).astype(np.float32)
    noise_im = np.random.uniform(-1, 1, (Ny, Nx)).astype(np.float32)
    return (amp*noise_amplitude*noise_re + 1j*amp*noise_amplitude*noise_im).astype(dtype)

def calculate_spectrum_log(field):
    """
    Calculate the spatial power spectrum in log scale:
        log10( |FFT(field)|^2 + 1e-6 )
    This helps reveal smaller peaks around the ring or hex spots.
    """
    spec = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2
    return np.log10(spec + 1e-6)

def setup_output_directories(output_directory):
    """
    Create directories for output if they don't exist.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plot_directory = os.path.join(output_directory, 'plots')
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    return plot_directory

def setup_real_time_viz(Nx, Ny, x_phys, y_phys, kx, ky, k_zoom_factor=1.5):
    """
    Initializes a figure with 3 subplots: amplitude, phase, log-spectrum.
    We artificially "zoom out" in the k-space extent by k_zoom_factor.
    """
    plt.ion()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Amplitude
    im_amp = ax[0].imshow(np.zeros((Ny, Nx), dtype=np.float32),
                          origin='lower',
                          extent=[x_phys.min(), x_phys.max(),
                                  y_phys.min(), y_phys.max()],
                          cmap='viridis')
    fig.colorbar(im_amp, ax=ax[0], label='Amplitude')
    ax[0].set_title('Amplitude')

    # Phase
    im_phase = ax[1].imshow(np.zeros((Ny, Nx), dtype=np.float32),
                            origin='lower',
                            extent=[x_phys.min(), x_phys.max(),
                                    y_phys.min(), y_phys.max()],
                            cmap='hsv', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im_phase, ax=ax[1], label='Phase')
    ax[1].set_title('Phase')

    # Spectrum (log scale) - with "zoomed-out" extent
    kx_min, kx_max = kx.min(), kx.max()
    ky_min, ky_max = ky.min(), ky.max()

    # Multiply each bound by k_zoom_factor to expand the displayed region
    kx_min *= k_zoom_factor
    kx_max *= k_zoom_factor
    ky_min *= k_zoom_factor
    ky_max *= k_zoom_factor

    im_spec = ax[2].imshow(
        np.zeros((Ny, Nx), dtype=np.float32),
        origin='lower',
        extent=[kx_min, kx_max, ky_min, ky_max],
        cmap='magma'
    )
    fig.colorbar(im_spec, ax=ax[2], label='Log10 Power')
    ax[2].set_title('Power Spectrum (log scale)')

    fig.tight_layout()
    plt.pause(0.01)
    return fig, (im_amp, im_phase, im_spec)

def update_real_time_viz(fig, im_handles, field, t):
    """
    Update amplitude, phase, and log-spectrum plots for the given field.
    field is in real space.
    """
    im_amp, im_phase, im_spec = im_handles

    amp = np.abs(field)
    phs = np.angle(field)
    log_spec = calculate_spectrum_log(field)  # log scale

    # Update amplitude
    im_amp.set_data(amp)
    im_amp.set_clim(0, amp.max())

    # Update phase
    im_phase.set_data(phs)
    # Phase is typically [-π, π], so we can leave the clim alone
    # but if you want to re-scale: im_phase.set_clim(-np.pi, np.pi)

    # Update log-spectrum
    im_spec.set_data(log_spec)
    # Colour range for log scale
    im_spec.set_clim(-3, 1)

    fig.suptitle(f"t = {t:.2f}")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def save_field_plots(A, directory, filename, x_phys, y_phys, kx, ky, k_zoom_factor=1.5):
    """
    Save amplitude, phase, and log-spectrum plots for field A (real space).
    Also "zoom out" for the k-space extent in the saved figure.
    """
    amp = np.abs(A)
    phs = np.angle(A)
    log_spec = calculate_spectrum_log(A)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Amplitude
    im1 = axes[0].imshow(amp, origin='lower',
                         extent=[x_phys.min(), x_phys.max(),
                                 y_phys.min(), y_phys.max()],
                         cmap='viridis')
    axes[0].set_title('Amplitude')
    plt.colorbar(im1, ax=axes[0], label='Amplitude')

    # Phase
    im2 = axes[1].imshow(phs, origin='lower', cmap='hsv',
                         extent=[x_phys.min(), x_phys.max(),
                                 y_phys.min(), y_phys.max()],
                         vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Phase')
    plt.colorbar(im2, ax=axes[1], label='Phase')

    # Log-Spectrum with zoomed k-extent
    kx_min, kx_max = kx.min()*k_zoom_factor, kx.max()*k_zoom_factor
    ky_min, ky_max = ky.min()*k_zoom_factor, ky.max()*k_zoom_factor

    im3 = axes[2].imshow(log_spec, origin='lower',
                         extent=[kx_min, kx_max, ky_min, ky_max],
                         cmap='magma')
    axes[2].set_title('Power Spectrum (log10 scale)')
    plt.colorbar(im3, ax=axes[2], label='Log10 Power')
    im3.set_clim(-3, 1)

    plt.suptitle(filename)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{filename}.png"), dpi=150)
    plt.close(fig)

def save_simulation_info(output_directory, params, finished=False):
    """
    Save simulation parameters and timing info to a text file.
    """
    info_file = os.path.join(output_directory, "simulation_info.txt")
    with open(info_file, 'a') as f:
        if not finished:
            f.write("\nKerr Cavity Simulation (START)\n")
            f.write("="*50 + "\n")
            f.write(f"Time started: {datetime.now()}\n")
            f.write(f"Parameters:\n")
            for k, v in params.items():
                f.write(f"  {k} = {v}\n")
            f.write("\n")
        else:
            f.write("\nKerr Cavity Simulation (END)\n")
            f.write("="*50 + "\n")
            f.write(f"Time finished: {datetime.now()}\n\n")

###############################################################################
# Main Simulation
###############################################################################
def main():
    # -----------------------------
    # Simulation & Physical Params
    # -----------------------------
    final_tau = 2000.0      # total sim time
    dt = 0.005
    theta = 1.0
    Is = 1.505
    beta_K = 2/3

    # Pump amplitude
    F_amp = np.sqrt(Is*(1.0 + (theta - beta_K*Is)**2))
    print(f"Calculated pump amplitude = {F_amp:.4f}")

    # Large domain
    Nx, Ny = 256, 256
    Lx, Ly = 50.0, 50.0  # Keep large domain
    dx, dy = Lx / Nx, Ly / Ny

    # Real-time viz and saving
    real_time_viz = True
    viz_update_freq = 200      # Update plots every 50 steps
    save_interval = 200.0     # Save data every 100 time units
    output_directory = "Kerr_Hexagonal_Patterns_LargeDomain"

    # -----------------------------
    # Set up directories & logging
    # -----------------------------
    plot_dir = setup_output_directories(output_directory)
    params = {
        "theta": theta,
        "Is": Is,
        "beta_K": beta_K,
        "F_amp": F_amp,
        "Nx, Ny": f"{Nx}, {Ny}",
        "Lx, Ly": f"{Lx}, {Ly}",
        "dt": dt,
        "final_tau": final_tau
    }
    # Clear or append info file
    open(os.path.join(output_directory, "simulation_info.txt"), 'w').close()
    save_simulation_info(output_directory, params, finished=False)

    # -----------------------------
    # Spatial Grid, Wavenumbers
    # -----------------------------
    x = np.linspace(-Lx/2, Lx/2, Nx).astype(np.float32)
    y = np.linspace(-Ly/2, Ly/2, Ny).astype(np.float32)
    X, Y = np.meshgrid(x, y, indexing='xy')

    kx = 2.0*np.pi*np.fft.fftfreq(Nx, d=dx).astype(np.float32)
    ky = 2.0*np.pi*np.fft.fftfreq(Ny, d=dy).astype(np.float32)
    Kx, Ky = np.meshgrid(kx, ky, indexing='xy')
    K_sq = (Kx**2 + Ky**2).astype(np.float32)

    # -----------------------------
    # Initial Field
    # -----------------------------
    # Start near zero but add small noise
    A0 = generate_noise(Nx, Ny, amp=0.1, noise_amplitude=1.0,
                        dtype=np.complex64)

    # Constant forcing
    pump = np.complex64(F_amp)

    # -----------------------------
    # Time Stepping Prep
    # -----------------------------
    A_fft = np.fft.fft2(A0)

    # Precompute half-step linear operator in k-space
    decay_phase = np.exp(-0.5*dt) * np.exp(-1j*(theta + K_sq)*dt/2).astype(np.complex64)

    current_tau = 0.0
    iteration = 0

    # Real-time Visualization
    # We pass in a "zoom factor" so the k-plot is artificially expanded
    k_zoom_factor = 1.25  # adjust as you like
    if real_time_viz:
        fig, im_handles = setup_real_time_viz(Nx, Ny, x, y, kx, ky,
                                              k_zoom_factor=k_zoom_factor)

    # Save initial state
    A_init = np.fft.ifft2(A_fft)
    save_field_plots(A_init, plot_dir, "initial_state", x, y, kx, ky,
                     k_zoom_factor=k_zoom_factor)


    # Main Loop

    next_save_time = save_interval
    while current_tau < final_tau:
        # 1) Half-step linear in k-space
        A_fft *= decay_phase

        # 2) Transform to real space
        A = np.fft.ifft2(A_fft)

        # 3) Nonlinear + Pump via 4th-order Runge-Kutta
        absA2 = np.abs(A)**2
        k1 = 1j*beta_K * absA2 * A + pump

        A_temp = A + 0.5*dt*k1
        absA2_temp = np.abs(A_temp)**2
        k2 = 1j*beta_K * absA2_temp * A_temp + pump

        A_temp = A + 0.5*dt*k2
        absA2_temp = np.abs(A_temp)**2
        k3 = 1j*beta_K * absA2_temp * A_temp + pump

        A_temp = A + dt*k3
        absA2_temp = np.abs(A_temp)**2
        k4 = 1j*beta_K * absA2_temp * A_temp + pump

        A = A + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # 4) Back to k-space
        A_fft = np.fft.fft2(A)

        # 5) Second half-step linear
        A_fft *= decay_phase

        # Update time
        current_tau += dt
        iteration += 1

        # Check for blow-up
        maxA = np.max(np.abs(A))
        if (np.isnan(maxA)) or (maxA > 1e6):
            print(f"Instability detected at t = {current_tau:.3f}. Aborting.")
            break

        # Print progress occasionally
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, t = {current_tau:.1f}/{final_tau}, max|A| = {maxA:.4f}")

        # Real-time Viz
        if real_time_viz and (iteration % viz_update_freq == 0):
            A_display = np.fft.ifft2(A_fft)
            update_real_time_viz(fig, im_handles, A_display, current_tau)
            # small pause for GUI
            plt.pause(0.001)

        # Save field snapshot at intervals
        if current_tau >= next_save_time:
            A_save = np.fft.ifft2(A_fft)
            save_field_plots(A_save, plot_dir, f"state_t{current_tau:.2f}",
                             x, y, kx, ky, k_zoom_factor=k_zoom_factor)
            next_save_time += save_interval

    # Final Outputs
    A_final = np.fft.ifft2(A_fft)
    save_field_plots(A_final, plot_dir, "final_state", x, y, kx, ky,
                     k_zoom_factor=k_zoom_factor)

    if real_time_viz:
        plt.ioff()  # Turn off interactive

    save_simulation_info(output_directory, params, finished=True)

    print(f"\nSimulation complete. Final time = {current_tau:.2f}")
    print(f"Results in: {output_directory}/")
    print(f"Max amplitude in final field = {np.max(np.abs(A_final)):.4f}")

    return A_final

if __name__ == "__main__":
    main()
