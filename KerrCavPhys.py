#!/usr/bin/env python3
"""
Kerr Cavity Simulator with Structured Light Pump Options & Physical Parameter Scaling

Split-step 2D Lugiato-Lefever (fixed dt).
Uses physical parameters to derive dimensionless simulation parameters.
Allows for Gaussian, Laguerre-Gaussian (LG), and Top-hat pump profiles.
Includes optional optical saturation.
Allows pump amplitude to be set based on a target dimensionless steady-state intensity (Is).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.special import assoc_laguerre # For LG modes
import scipy.constants as const

###############################################################################
# Physical Constants
###############################################################################
c_light = const.c           # Speed of light [m/s]
hbar = const.hbar           # Reduced Planck constant [J·s]
epsilon_0 = const.epsilon_0 # Vacuum permittivity [F/m]

###############################################################################
# Physical to Dimensionless Parameter Functions for LLE
###############################################################################

def calculate_dimensionless_lle_parameters(physical_params_lle):
    """
    Calculate dimensionless parameters for the Lugiato-Lefever Equation (LLE)
    from physical inputs.

    Input physical_params_lle dict should contain:
    - 'wavelength_m': Optical pump wavelength [m]
    - 'w_s_um': Scaling beam waist [μm]. This waist w_s defines the dimensionless
                  transverse coordinates x' = x_phys / w_s.
    - 'cavity_length_m': Total cavity round-trip length (L_cav) [m]
    - 'T_mirror': Mirror intensity transmission coefficient
    - 'cavity_detuning_MHz': Pump-cavity resonance detuning (omega_Pump - omega_Cavity) [MHz]
    """
    p = physical_params_lle

    # Convert units
    w_s_meters = p['w_s_um'] * 1e-6  # Scaling waist to meters
    delta_omega_cavity_rad_s = 2 * np.pi * p['cavity_detuning_MHz'] * 1e6  # (omega_P - omega_c) in rad/s

    # Derived quantities
    k_L_pump = 2 * np.pi / p['wavelength_m']  # Optical pump wavevector [m^-1]

    # Kappa - Cavity field decay rate [rad/s]. Defines the dimensionless time tau = kappa * t_physical.
    # The factor of 2 in 2*L_cav is for a ring cavity; for Fabry-Perot, it might differ based on definition.
    kappa_timescale = (c_light * p['T_mirror']) / (2 * p['cavity_length_m'])

    # Theta - Dimensionless cavity detuning.
    # Standard LLE: dA/dtau = PUMP - (1 + i*theta)A + ...
    # where theta = (omega_c - omega_P) / kappa.
    # Given delta_omega_cavity_rad_s = (omega_P - omega_c):
    theta_dimless = -delta_omega_cavity_rad_s / kappa_timescale

    # Alpha - Dimensionless optical diffraction coefficient.
    # For the LLE term i*alpha*nabla_perp^2 A, where nabla_perp^2 acts on x' = x_phys/w_s.
    # alpha = (2 * L_cav) / (k_L_pump * w_s_meters^2 * T_mirror)
    # This makes alpha the ratio of cavity decay length to diffraction length (scaled by w_s).
    # For the standard LLE form (i * nabla^2 A), we want this alpha_diffraction to be 1.0.
    # This is achieved by choosing w_s appropriately.
    alpha_diffraction = (2 * p['cavity_length_m']) / (k_L_pump * w_s_meters**2 * p['T_mirror'])
    
    # Suggested dimensionless time step based on a physical time step (e.g., 10 picoseconds)
    # This is just a rough guide.
    suggested_dt_dimensionless = (10e-12) * kappa_timescale

    return {
        'theta': theta_dimless,
        'alpha_diffraction': alpha_diffraction,
        'kappa_timescale': kappa_timescale,
        'suggested_dt': suggested_dt_dimensionless,
        'w_s_meters': w_s_meters,
        'k_L_pump_inv_meters': k_L_pump
    }

###############################################################################
# Utility & Simulation Functions
###############################################################################

def generate_noise(Nx, Ny, amp, noise_amplitude, dtype=np.complex64):
    """Generates complex random noise."""
    noise_re = np.random.uniform(-1, 1, (Ny, Nx)).astype(np.float32)
    noise_im = np.random.uniform(-1, 1, (Ny, Nx)).astype(np.float32)
    return (amp * noise_amplitude * noise_re + 1j * amp * noise_amplitude * noise_im).astype(dtype)

def calculate_spectrum_log(field):
    """Calculates log10 of the power spectrum, shifted for visualization."""
    # FFT and then shift DC to center for spectrum calculation
    spec = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2
    return np.log10(spec + 1e-9) # Epsilon to avoid log(0)

def setup_output_directories(output_directory):
    """Creates output directories if they don't exist."""
    if not os.path.exists(output_directory): os.makedirs(output_directory)
    plot_dir = os.path.join(output_directory, 'plots')
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    return plot_dir

def setup_real_time_viz(Nx, Ny, x_phys_um, y_phys_um, kx_plot_shifted_um_inv, ky_plot_shifted_um_inv, k_zoom_factor=1.5):
    """Initializes matplotlib figure for real-time visualization."""
    plt.ion(); fig, ax = plt.subplots(1, 3, figsize=(18, 6)) # rows, cols
    # Physical space extents for plotting
    real_extent = [x_phys_um.min(), x_phys_um.max(), y_phys_um.min(), y_phys_um.max()]
    # K-space extents for plotting, with zoom
    kx_min_p, kx_max_p = kx_plot_shifted_um_inv.min()*k_zoom_factor, kx_plot_shifted_um_inv.max()*k_zoom_factor
    ky_min_p, ky_max_p = ky_plot_shifted_um_inv.min()*k_zoom_factor, ky_plot_shifted_um_inv.max()*k_zoom_factor
    k_extent = [kx_min_p, kx_max_p, ky_min_p, ky_max_p]

    # Amplitude plot
    im_amp = ax[0].imshow(np.zeros((Ny,Nx)), origin='lower', extent=real_extent, cmap='viridis')
    fig.colorbar(im_amp,ax=ax[0],label='|A|'); ax[0].set_title('Amplitude'); ax[0].set_xlabel('x [μm]'); ax[0].set_ylabel('y [μm]')
    # Phase plot
    im_phase = ax[1].imshow(np.zeros((Ny,Nx)), origin='lower', extent=real_extent, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im_phase,ax=ax[1],label='Phase [rad]'); ax[1].set_title('Phase'); ax[1].set_xlabel('x [μm]'); ax[1].set_ylabel('y [μm]')
    # Spectrum plot
    im_spec = ax[2].imshow(np.zeros((Ny,Nx)), origin='lower', extent=k_extent, cmap='magma')
    fig.colorbar(im_spec,ax=ax[2],label='Log10 Power'); ax[2].set_title('Spectrum (DC at center)'); ax[2].set_xlabel('k$_x$ [μm$^{-1}$]'); ax[2].set_ylabel('k$_y$ [μm$^{-1}$]')
    
    fig.tight_layout(); plt.pause(0.01)
    return fig, (im_amp, im_phase, im_spec)

def update_real_time_viz(fig, im_handles, field_real_space, t_tau):
    """Updates the real-time visualization plots."""
    im_amp, im_phase, im_spec = im_handles
    amp = np.abs(field_real_space)
    phs = np.angle(field_real_space)
    log_spec = calculate_spectrum_log(field_real_space) # field_real_space is FFTd then shifted inside
    
    amp_max = amp.max() if amp.size > 0 and amp.max() > 1e-9 else 1.0
    log_spec_min = log_spec.min() if log_spec.size > 0 else -9.0 # Adjusted min for log_spec
    log_spec_max = log_spec.max() if log_spec.size > 0 and log_spec.max() > log_spec_min + 1e-9 else log_spec_min + 1.0

    im_amp.set_data(amp); im_amp.set_clim(0, amp_max)
    im_phase.set_data(phs) # Phase clim is fixed [-pi, pi]
    im_spec.set_data(log_spec); im_spec.set_clim(log_spec_min, log_spec_max)
    
    fig.suptitle(f"Dimensionless Time τ = {t_tau:.2f}"); fig.canvas.draw_idle(); fig.canvas.flush_events()

def save_field_plots(A_real_space, directory, filename_prefix, x_phys_um, y_phys_um, kx_plot_shifted_um_inv, ky_plot_shifted_um_inv, k_zoom_factor=1.5):
    """Saves plots of amplitude, phase, and spectrum to a file."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    real_extent = [x_phys_um.min(), x_phys_um.max(), y_phys_um.min(), y_phys_um.max()]
    kx_min_p = kx_plot_shifted_um_inv.min()*k_zoom_factor; kx_max_p = kx_plot_shifted_um_inv.max()*k_zoom_factor
    ky_min_p = ky_plot_shifted_um_inv.min()*k_zoom_factor; ky_max_p = ky_plot_shifted_um_inv.max()*k_zoom_factor
    k_extent = [kx_min_p, kx_max_p, ky_min_p, ky_max_p]
    
    amp = np.abs(A_real_space)
    phs = np.angle(A_real_space)
    log_spec = calculate_spectrum_log(A_real_space) # field_real_space is FFTd then shifted inside

    amp_max = amp.max() if amp.size > 0 and amp.max() > 1e-9 else 1.0
    log_spec_min = log_spec.min() if log_spec.size > 0 else -9.0
    log_spec_max = log_spec.max() if log_spec.size > 0 and log_spec.max() > log_spec_min + 1e-9 else log_spec_min + 1.0

    im0=axes[0].imshow(amp, origin='lower',extent=real_extent,cmap='viridis'); axes[0].set_title('Amplitude'); axes[0].set_xlabel('x [μm]'); axes[0].set_ylabel('y [μm]'); fig.colorbar(im0,ax=axes[0],label='|A|'); im0.set_clim(0, amp_max)
    im1=axes[1].imshow(phs, origin='lower',extent=real_extent,cmap='hsv',vmin=-np.pi,vmax=np.pi); axes[1].set_title('Phase'); axes[1].set_xlabel('x [μm]'); axes[1].set_ylabel('y [μm]'); fig.colorbar(im1,ax=axes[1],label='Phase [rad]')
    im2=axes[2].imshow(log_spec, origin='lower',extent=k_extent,cmap='magma'); axes[2].set_title('Spectrum (DC at center)'); axes[2].set_xlabel('k$_x$ [μm$^{-1}$]'); axes[2].set_ylabel('k$_y$ [μm$^{-1}$]'); fig.colorbar(im2,ax=axes[2],label='Log10 Power'); im2.set_clim(log_spec_min, log_spec_max)
    
    plt.suptitle(filename_prefix.replace("_"," ").title()); plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(directory,f"{filename_prefix}.png"),dpi=150); plt.close(fig)

def save_simulation_info(output_directory, params_dict, finished=False):
    """Saves simulation parameters and start/end times to a log file."""
    info_file = os.path.join(output_directory, "simulation_info.txt")
    write_header = not os.path.exists(info_file) or os.path.getsize(info_file) == 0
    
    with open(info_file, 'a') as f:
        if write_header and not finished: # Write header only for a brand new simulation run
            f.write("\nKerr Cavity Simulation with Physical Params (START)\n"); f.write("="*50 + "\n")
            f.write(f"Time started: {datetime.now()}\n"); f.write("Parameters:\n")
            for k, v in params_dict.items(): f.write(f"  {k} = {v}\n")
            f.write("\n")
        elif finished:
            if write_header: # Safeguard if simulation aborted before any info was logged
                 f.write("\nKerr Cavity Simulation with Physical Params (INFO INCOMPLETE)\n"); f.write("="*50 + "\n")
                 f.write(f"Time started: {datetime.now()} (approx)\n"); f.write("Parameters (may be incomplete):\n")
                 for k, v in params_dict.items(): f.write(f"  {k} = {v}\n")
                 f.write("\n")
            f.write("\nKerr Cavity Simulation (END)\n"); f.write("="*50 + "\n")
            f.write(f"Time finished: {datetime.now()}\n\n")

# Structured Pump Generation Functions
def generate_lg_pump(X_um, Y_um, F_amp_dimless, w0_phys_um, l_oam, p_radial, noise_amplitude_factor=0.001):
    """Generates a Laguerre-Gaussian pump profile in physical units."""
    r_sq_um = X_um**2 + Y_um**2; w0_phys_um_sq = w0_phys_um**2
    if w0_phys_um_sq < 1e-12: # Check for very small waist
        print("Warning: LG pump waist w0 is near zero. Returning zero field.")
        return np.zeros_like(X_um,dtype=np.complex64)
    
    radial_norm_sq = r_sq_um / w0_phys_um_sq # (r/w0)^2
    phi_rad = np.arctan2(Y_um, X_um)
    
    # scipy.special.assoc_laguerre(x, order, degree) is L_order^degree(x)
    # For LG_p^l (p radial, l azimuthal), order=p, degree=|l|
    # Argument of Laguerre polynomial is 2*r^2/w0^2 = 2*radial_norm_sq
    laguerre_poly = assoc_laguerre(2 * radial_norm_sq, p_radial, np.abs(l_oam))
    
    # Radial part of LG mode: (sqrt(2)*r/w0)^|l| * L_p^|l|(2r^2/w0^2) * exp(-r^2/w0^2)
    # (sqrt(2)*r/w0)^|l| = (sqrt(2*radial_norm_sq))^|l|
    radial_factor = (np.sqrt(2 * radial_norm_sq))**np.abs(l_oam)
    
    lg_profile = radial_factor * laguerre_poly * np.exp(-radial_norm_sq) * np.exp(1j * l_oam * phi_rad)
    
    # Normalize the spatial profile so its peak absolute value is 1
    # This makes F_amp_dimless the actual peak amplitude of the pump.
    max_abs_spatial_profile = np.max(np.abs(lg_profile))
    if max_abs_spatial_profile > 1e-9: # Avoid division by zero for null profiles (e.g. l=0, p>0 at r=0)
        norm_lg = F_amp_dimless * (lg_profile / max_abs_spatial_profile)
    else: # If profile is essentially zero, keep it zero before adding noise
        norm_lg = np.zeros_like(lg_profile, dtype=np.complex64)
        
    # Add small noise to the pump
    noise = generate_noise(X_um.shape[1], X_um.shape[0], F_amp_dimless, noise_amplitude_factor, np.complex64)
    return (norm_lg + noise).astype(np.complex64)

def generate_tophat_pump(X_um, Y_um, F_amp_dimless, radius_phys_um, steepness, noise_amplitude_factor=0.001, add_oam=False, oam_l=0):
    """Generates a top-hat pump profile with optional OAM."""
    r_um = np.sqrt(X_um**2 + Y_um**2)
    # tanh provides smooth edges. Larger steepness -> sharper edge.
    profile_real = F_amp_dimless * 0.5 * (1 - np.tanh(steepness * (r_um - radius_phys_um)))
    
    if add_oam and oam_l != 0:
        phi_rad = np.arctan2(Y_um, X_um)
        final_profile = profile_real * np.exp(1j * oam_l * phi_rad)
    else:
        final_profile = profile_real
        
    noise = generate_noise(X_um.shape[1],X_um.shape[0],F_amp_dimless,noise_amplitude_factor,np.complex64)
    return (final_profile + noise).astype(np.complex64)

###############################################################################
# Main Simulation
###############################################################################
def main():
    # -----------------------------
    # Define Physical Parameters for LLE
    # -----------------------------
    physical_params_lle = {
        'wavelength_m': 720e-9,      # Pump laser wavelength
        'w_s_um': 478.7,              # CRITICAL: Scaling beam waist [μm] for alpha_diffraction ≈ 1
        'cavity_length_m': 10e-3,     # Cavity round-trip length (L_cav) [m]
        'T_mirror': 0.01,             # Mirror intensity transmission coefficient
        'cavity_detuning_MHz': -23.85,# CRITICAL: For theta_dimless ≈ 1.0 (Pump freq < Cavity freq)
    }
    
    # Calculate dimensionless parameters based on physical inputs
    calc_params_lle = calculate_dimensionless_lle_parameters(physical_params_lle)
    
    theta_dimless = calc_params_lle['theta']
    alpha_diffraction_dimless = calc_params_lle['alpha_diffraction'] # This should be ~1.0 now
    kappa_timescale_rad_s = calc_params_lle['kappa_timescale']
    
    # -----------------------------
    # Directly Set Dimensionless LLE Parameters
    # -----------------------------
    beta_K_dimless = 2/3       # Dimensionless Kerr coefficient (standard value)
    sigma_sat_dimless = 0.0    # Dimensionless saturation intensity (0 for no saturation)
    
    # Target dimensionless steady-state intensity (for plane-wave pump reference)
    Is_target_dimless = 1.505  # Standard value for pattern formation studies
    
    # Calculate F_amp based on Is_target_dimless for a plane-wave steady state
    # Formula: F_amp^2 = Is * (1 + (theta - beta_K * Is)^2)
    F_amp_calculated_from_Is = np.sqrt(Is_target_dimless * (1 + (theta_dimless - beta_K_dimless * Is_target_dimless)**2))
    print(f"F_amp calculated from Is={Is_target_dimless:.3f}, theta={theta_dimless:.3f}, beta_K={beta_K_dimless:.3f} -> F_amp = {F_amp_calculated_from_Is:.4f}")

    # Option to manually override F_amp or use the one calculated from Is
    F_amp_dimless_manual_override = None # Set to a numerical value to override, else None
    
    if F_amp_dimless_manual_override is not None:
        F_amp_dimless_used = F_amp_dimless_manual_override
        print(f"Using manually overridden F_amp_dimless: {F_amp_dimless_used:.4f}")
    else:
        F_amp_dimless_used = F_amp_calculated_from_Is
        print(f"Using F_amp_dimless calculated from Is: {F_amp_dimless_used:.4f}")

    # --- Print derived dimensionless parameters for verification ---
    print("\n=== Derived Dimensionless LLE Parameters ===")
    print(f"theta (dimensionless detuning) = {theta_dimless:.4f}")
    print(f"alpha_diffraction (dimless diff. coeff.) = {alpha_diffraction_dimless:.4f}") # Should be ~1.0
    print(f"kappa_timescale = {kappa_timescale_rad_s:.2e} rad/s (defines 1 unit of tau)")
    print(f"--- Directly Set/Calculated Dimensionless ---")
    print(f"beta_K (dimless Kerr coeff.) = {beta_K_dimless:.4f}")
    print(f"sigma_sat (dimless saturation) = {sigma_sat_dimless:.4f}")
    print(f"Is_target (for F_amp calc) = {Is_target_dimless:.4f}")
    print(f"F_amp_peak_used (dimless pump peak) = {F_amp_dimless_used:.4f}")
    print(f"Suggested dimensionless dt (for dt_phys=10ps) = {calc_params_lle['suggested_dt']:.2e}")
    print("=========================================\n")

    # -----------------------------
    # Simulation Time & Grid
    # -----------------------------
    final_tau = 200.0 # Total dimensionless simulation time (can be reduced for faster tests)
    dt_dimless = 0.005 # Dimensionless time step. Smaller can be more accurate but slower.
                       # Original KerrCavity.py used dt=0.1, which was too large for its grid.
                       # dt=0.005 here with Nx=256, Lx_dimless=50 means max(K_sq*dt/2) ~ (pi/dx_dimless)^2 * dt/2
                       # dx_dimless = 50/256 ~ 0.195. max_K_dimless ~ pi/0.195 ~ 16. max_K_sq ~ 256.
                       # max_K_sq*alpha_diff*dt/2 ~ 256*1*0.005/2 ~ 0.64 radians. This is acceptable.
    
    Nx, Ny = 256, 256             # Number of grid points
    # Physical domain size [μm]. Should correspond to a reasonable dimensionless domain.
    # Lx_dimless = Lx_um / w_s_um. If w_s_um=478.7um and Lx_dimless=50, then Lx_um = 50*478.7 = 23935um
    Lx_um, Ly_um = 24000.0, 24000.0 # CRITICAL: For Lx_dimless ≈ 50
    dx_um, dy_um = Lx_um / Nx, Ly_um / Ny # Physical grid spacing [μm]

    # -----------------------------
    # Pump Configuration (using physical units for waists/radii)
    # -----------------------------
    pump_type = "gaussian"         # "gaussian" (homogeneous), "LG", or "tophat"
    # These are only used if pump_type is "LG" or "tophat"
    pump_waist_um = Lx_um / 5.0  # Physical waist for LG (e.g. 1/5th of domain) or radius for tophat
    pump_lg_l = 1; pump_lg_p = 0 # Example LG mode parameters
    pump_tophat_steepness = 20.0 / (pump_waist_um/10) # Example steepness for tophat
    pump_tophat_add_oam = False; pump_tophat_oam_l = 0
    noise_on_pump_factor = 0.001 # Relative amplitude of noise on pump

    # -----------------------------
    # Output and Visualization
    # -----------------------------
    real_time_viz = True
    viz_update_freq = 200 # Update visualization every N iterations
    save_interval_tau = 20.0 # Save plots every N units of tau
    k_plot_zoom_factor = 1.5 # Zoom factor for k-space plots for better visibility

    output_directory_base = f"KerrCavity_Phys_Pump_{pump_type}"
    if pump_type == "LG": output_directory_base += f"_l{pump_lg_l}p{pump_lg_p}"
    # Add key dimensionless params to folder name for easier identification
    output_directory_base += f"_th{theta_dimless:.2f}_Is{Is_target_dimless:.2f}_F{F_amp_dimless_used:.2f}".replace('.','p')
    
    plot_dir = setup_output_directories(output_directory_base)
    
    # Log parameters
    params_to_log = {
        "Physical: Wavelength": f"{physical_params_lle['wavelength_m']*1e9:.1f} nm",
        "Physical: Scaling Waist (w_s)": f"{physical_params_lle['w_s_um']:.1f} μm",
        "Physical: Cavity Length (L_cav)": f"{physical_params_lle['cavity_length_m']*1e3:.1f} mm",
        "Physical: Mirror Transmission (T)": f"{physical_params_lle['T_mirror']:.3f}",
        "Physical: Cavity Detuning (omega_P-omega_c)": f"{physical_params_lle['cavity_detuning_MHz']:.2f} MHz",
        "--- Dimensionless Derived ---": "--- ---",
        "theta (detuning)": f"{theta_dimless:.4f}", 
        "alpha_diffraction (diff. coeff.)": f"{alpha_diffraction_dimless:.4f}",
        "kappa_timescale (time unit)": f"{kappa_timescale_rad_s:.2e} rad/s",
        "--- Dimensionless Set ---": "--- ---",
        "Is_target (for F_amp calc)": f"{Is_target_dimless:.4f}",
        "F_amp_peak_used (pump peak)": f"{F_amp_dimless_used:.4f}",
        "beta_K (Kerr coeff.)": f"{beta_K_dimless:.4f}", 
        "sigma_sat (saturation)": f"{sigma_sat_dimless:.4f}",
        "--- Grid & Time ---": "--- ---",
        "Grid Pts (Nx, Ny)": f"{Nx}x{Ny}", 
        "Domain Size (Lx_um, Ly_um)": f"{Lx_um:.0f}x{Ly_um:.0f} μm",
        "Dimensionless Domain (Lx', Ly')": f"{Lx_um/physical_params_lle['w_s_um']:.2f}x{Ly_um/physical_params_lle['w_s_um']:.2f}",
        "dt (dimless time step)": f"{dt_dimless:.2e}", 
        "final_tau (total dimless time)": final_tau,
        "--- Pump ---": "--- ---",
        "Pump Profile Type": pump_type, 
        "Pump Waist/Radius (if structured)": f"{pump_waist_um:.1f} μm",
    }
    info_file_path = os.path.join(output_directory_base, "simulation_info.txt")
    if os.path.exists(info_file_path) and os.path.getsize(info_file_path) > 0 : # If file exists and is not empty, remove for fresh log
        os.remove(info_file_path) 
    save_simulation_info(output_directory_base, params_to_log, finished=False)

    # -----------------------------
    # Spatial Grid & Wavenumbers
    # -----------------------------
    # Physical grid (units of μm)
    x_um = np.linspace(-Lx_um/2, Lx_um/2, Nx, endpoint=False).astype(np.float32)
    y_um = np.linspace(-Ly_um/2, Ly_um/2, Ny, endpoint=False).astype(np.float32)
    X_um, Y_um = np.meshgrid(x_um, y_um, indexing='xy')

    # Physical wavenumbers (units of μm⁻¹)
    # np.fft.fftfreq provides unshifted frequencies: DC at index 0, then positive, then negative.
    kx_um_inv = 2.0*np.pi*np.fft.fftfreq(Nx, d=dx_um).astype(np.float32)
    ky_um_inv = 2.0*np.pi*np.fft.fftfreq(Ny, d=dy_um).astype(np.float32)
    # Create 2D grids for kx, ky (still unshifted, DC at [0,0] for K_sq_um_inv)
    Kx_um_inv_grid, Ky_um_inv_grid = np.meshgrid(kx_um_inv, ky_um_inv, indexing='xy') 
    
    # Dimensionless squared wavenumber for the LLE operator
    # K_dimless = K_physical * w_s. So K_sq_dimless = K_sq_physical * w_s^2.
    w_s_um = physical_params_lle['w_s_um'] # Physical scaling waist in μm
    # K_sq_dimless will have its DC component at index [0,0] because Kx_um_inv_grid, Ky_um_inv_grid do.
    K_sq_dimless = (Kx_um_inv_grid**2 + Ky_um_inv_grid**2) * (w_s_um**2) 
    
    # For plotting k-space with DC at the center:
    kx_plot_shifted_um_inv = np.fft.fftshift(kx_um_inv)
    ky_plot_shifted_um_inv = np.fft.fftshift(ky_um_inv)

    # -----------------------------
    # Pump Field (Dimensionless Amplitude, F_amp_dimless_used is its peak)
    # -----------------------------
    if pump_type == "LG":
        F_P_dimless = generate_lg_pump(X_um, Y_um, F_amp_dimless_used, pump_waist_um, pump_lg_l, pump_lg_p, noise_on_pump_factor)
    elif pump_type == "tophat":
        F_P_dimless = generate_tophat_pump(X_um, Y_um, F_amp_dimless_used, pump_waist_um, pump_tophat_steepness, noise_on_pump_factor, pump_tophat_add_oam, pump_tophat_oam_l)
    else: # "gaussian" (homogeneous) pump
        F_P_dimless = np.full((Ny, Nx), F_amp_dimless_used, dtype=np.complex64)
        # Add a small amount of noise to the homogeneous pump as well
        F_P_dimless += generate_noise(Nx,Ny,F_amp_dimless_used,noise_on_pump_factor,np.complex64)

    # Initial Cavity Field (Dimensionless Amplitude)
    # Start with small random noise, scaled relative to the pump amplitude
    A0_dimless = generate_noise(Nx, Ny, amp=0.01 * F_amp_dimless_used, noise_amplitude=1.0)

    # -----------------------------
    # Time Stepping Prep
    # -----------------------------
    A_dimless = A0_dimless.copy()
    A_fft = np.fft.fft2(A_dimless) # A_fft has DC component at index [0,0]

    # Linear Operator for LLE in Fourier Space.
    # The LLE is dA/dtau = PUMP - (1+i*theta)A + i*beta_K*|A|^2*A + i*alpha_diff*nabla_sq_dimless*A
    # Linear part in k-space (after FFT of A): -(1+i*theta)A_fft - i*alpha_diff*K_sq_dimless*A_fft
    # So, for split-step, the exponent is ( -(1+i*theta) - i*alpha_diff*K_sq_dimless )
    # CRITICAL FIX: Use K_sq_dimless (DC at [0,0]) to match A_fft convention.
    linear_operator_exponent = (-(1.0 + 1j*theta_dimless) - 1j*alpha_diffraction_dimless*K_sq_dimless)
    linear_op_half_dt = np.exp( (dt_dimless/2.0) * linear_operator_exponent ).astype(np.complex64)

    current_tau = 0.0
    iteration = 0
    if real_time_viz: # Setup visualization
        fig_viz, im_handles_viz = setup_real_time_viz(Nx, Ny, x_um, y_um, kx_plot_shifted_um_inv, ky_plot_shifted_um_inv, k_plot_zoom_factor)
    
    # Save initial state
    save_field_plots(A_dimless, plot_dir, f"state_tau{current_tau:.2f}", x_um, y_um, kx_plot_shifted_um_inv, ky_plot_shifted_um_inv, k_plot_zoom_factor)
    next_save_tau = save_interval_tau # Initialize next save time
    
    print("\nStarting Kerr cavity simulation loop...")
    # -----------------------------
    # Main Evolution Loop (Split-Step Fourier Method)
    # -----------------------------
    while current_tau < final_tau:
        # --- Step 1: Half linear step in Fourier space ---
        A_fft *= linear_op_half_dt       
        
        # --- Step 2: Full nonlinear step in real space (using RK4) ---
        A_dimless = np.fft.ifft2(A_fft) # Transform to real space
        
        # RK4 integration for: dA/dtau = F_P_dimless + (i*beta_K*|A|^2 / (1+sigma_sat*|A|^2)) * A
        # This is the nonlinear part of the LLE.
        
        # k1
        absA2_k1 = np.abs(A_dimless)**2
        denominator_sat_k1 = 1.0 + sigma_sat_dimless * absA2_k1
        if sigma_sat_dimless > 0: denominator_sat_k1 += 1e-12 # Avoid division by zero if sigma is active
        nonlinear_term_k1 = (1j * beta_K_dimless * absA2_k1) / denominator_sat_k1
        k1_rhs = F_P_dimless + nonlinear_term_k1 * A_dimless
        
        # k2
        A_temp_k2 = A_dimless + 0.5 * dt_dimless * k1_rhs
        absA2_k2 = np.abs(A_temp_k2)**2
        denominator_sat_k2 = 1.0 + sigma_sat_dimless * absA2_k2
        if sigma_sat_dimless > 0: denominator_sat_k2 += 1e-12
        nonlinear_term_k2 = (1j * beta_K_dimless * absA2_k2) / denominator_sat_k2
        k2_rhs = F_P_dimless + nonlinear_term_k2 * A_temp_k2

        # k3
        A_temp_k3 = A_dimless + 0.5 * dt_dimless * k2_rhs
        absA2_k3 = np.abs(A_temp_k3)**2
        denominator_sat_k3 = 1.0 + sigma_sat_dimless * absA2_k3
        if sigma_sat_dimless > 0: denominator_sat_k3 += 1e-12
        nonlinear_term_k3 = (1j * beta_K_dimless * absA2_k3) / denominator_sat_k3
        k3_rhs = F_P_dimless + nonlinear_term_k3 * A_temp_k3

        # k4
        A_temp_k4 = A_dimless + dt_dimless * k3_rhs
        absA2_k4 = np.abs(A_temp_k4)**2
        denominator_sat_k4 = 1.0 + sigma_sat_dimless * absA2_k4
        if sigma_sat_dimless > 0: denominator_sat_k4 += 1e-12
        nonlinear_term_k4 = (1j * beta_K_dimless * absA2_k4) / denominator_sat_k4
        k4_rhs = F_P_dimless + nonlinear_term_k4 * A_temp_k4
        
        # Combine RK4 steps
        A_dimless += (dt_dimless / 6.0) * (k1_rhs + 2*k2_rhs + 2*k3_rhs + k4_rhs)
        
        # --- Step 3: Second half linear step in Fourier space ---
        A_fft = np.fft.fft2(A_dimless) # Transform back to k-space
        A_fft *= linear_op_half_dt       
        
        # --- Update time and iteration ---
        current_tau += dt_dimless
        iteration += 1

        # --- Monitoring and Visualization ---
        if iteration % viz_update_freq == 0: # Check/print less frequently than every 100 if dt is small
            # For max|A| display, it's better to use the real-space field before the next FFT
            # However, A_dimless is already updated. If needed, can ifft(A_fft) for display.
            current_A_for_display = np.fft.ifft2(A_fft) # Get current field in real space
            maxA_abs_curr = np.max(np.abs(current_A_for_display)) 
            print(f"Iter {iteration}, τ={current_tau:.2f}/{final_tau:.0f}, Max|A|={maxA_abs_curr:.4f}")
            
            # Check for instability (field blowing up)
            if np.isnan(maxA_abs_curr) or maxA_abs_curr > 1e3: # Adjusted instability threshold
                print(f"Instability detected: Max|A| = {maxA_abs_curr:.2e}. Aborting.")
                save_field_plots(current_A_for_display, plot_dir, f"state_ABORTED_tau{current_tau:.2f}", x_um, y_um, kx_plot_shifted_um_inv, ky_plot_shifted_um_inv, k_plot_zoom_factor)
                break # Exit simulation loop
        
        if real_time_viz and (iteration % viz_update_freq == 0):
            if 'current_A_for_display' not in locals(): # Ensure it's defined if not printed above
                 current_A_for_display = np.fft.ifft2(A_fft)
            update_real_time_viz(fig_viz, im_handles_viz, current_A_for_display, current_tau)
            
        # --- Save field plots at intervals ---
        if current_tau >= next_save_tau:
            field_to_save = np.fft.ifft2(A_fft) # Get current field in real space
            save_field_plots(field_to_save, plot_dir, f"state_tau{current_tau:.2f}", x_um, y_um, kx_plot_shifted_um_inv, ky_plot_shifted_um_inv, k_plot_zoom_factor)
            next_save_tau += save_interval_tau # Increment for next save
            print(f"Saved snapshot at tau = {current_tau:.2f}")
            
    # --- End of Simulation Loop ---
    
    A_final_dimless = np.fft.ifft2(A_fft) # Final field in real space
    save_field_plots(A_final_dimless, plot_dir, "final_state", x_um, y_um, kx_plot_shifted_um_inv, ky_plot_shifted_um_inv, k_plot_zoom_factor)
    
    if real_time_viz: # Keep final plot open if viz was on
        update_real_time_viz(fig_viz, im_handles_viz, A_final_dimless, current_tau) # Show final state
        print("Real-time visualization finished. Close plot window to exit.")
        plt.ioff(); plt.show()
        
    save_simulation_info(output_directory_base, params_to_log, finished=True) # Log end of simulation
    print(f"\nSimulation complete. Final dimensionless time τ = {current_tau:.2f}")
    print(f"Max|A| in final state: {np.max(np.abs(A_final_dimless)):.4f}")
    print(f"Results saved in: {output_directory_base}/")

    return A_final_dimless

if __name__ == "__main__":
    plt.close('all') # Close any pre-existing matplotlib figures
    main()

