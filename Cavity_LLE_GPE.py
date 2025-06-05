# Grant W. Henderson, University of Strathclyde, grant.henderson@strath.ac.uk
# Basic SSFM to describe coupled ultracold atomic and optical beams in a driven optical cavity
# Model details: Coupled Lugiato-Lefever and Gross-Pitaevskii Equations.
# Last update: 29/01/2025

###############################################################################
# Parameter setup
###############################################################################

final_tau = 5 * 10**(3) # tau-value to evolve to

w_L = 20 # w_0, characteristic optical beam waist, microns
w_F = 50 # Optional control to change the pump's w_F without w_L, microns. In PRL, w_F = w_L.
l_mode = 2 # \ell-index of the optical pump
l_mode_prof = l_mode # Optional control to alter OAM without profile change
OAM = 1 # Do you want OAM on = 1? or off = 0?
p_mode = 0 # p-index of the optical pump
wavelength = 720 * 10**-9 # Optical beam wavelength
F_amp = 2.5 # Initial amplitude of optical pump

BEC_w_F = 200 # w_\psi, initial BEC beam waist, microns
Psi_amp = 1.0 # Initial BEC amplitude

noise_amplitude = 10**-2 # Relative to field amplitude

s = -1 # Sign of the BEC-field detuning, s=-1 (red), s=1 (blue)
atomic_prefac = 10**(-1) # Atomic field prefactor, alpha_psi / kappa
beta_col = 2.5 # BEC scattering parameter
L_3 = 0.00022 # BEC three body loss parameter
theta = 1.0 # cavity detuning
alpha_light = 1 # Numerical prefactor ahead of optical Laplacian
beta_light = 1 # Numerical prefactor ahead of optical nonlinear terms
sigma = 0.0 # Optical saturation parameter 

N_w = 40 # Number of characteristic beam waists in domain size, multiples of w_L
Nx = Ny = 2**8 # Number of grid points, square grid s.t. Nx = Ny
boundary_extent = 0.85 # proportion of grid covered by boundaries, 0 -> 1
boundary_grad = 1 # Gradient of absorbing boundaries, fairly arbitrary

data_fac = final_tau / 50 # tau-interval to store data at
plot_fac = final_tau / 50 # tau-interval to plot fields at
track_to_max = 1 # Do you want your field plots all normalised to run max? Yes = 1, No = 0

output_directory = 'SimulationOutput' # Name of directory you wish to store output files
overwrite = 1 # Do you want to overwrite results if 'output_directory' exists? Yes = 1, No = 0 (job will fail)

###############################################################################
# Packages
###############################################################################

import numpy
import FOAMilyCAC as CACFuncs
import sys

###############################################################################
# Grid setup and scalings
###############################################################################

k_L = (numpy.pi * 2) / wavelength # Optical wavenumber
xy_scaling = w_L / numpy.sqrt(2) # Transverse scaling
z_scaling = k_L * (w_L)**2 # Longitudinal scaling, corresponding to z = 2z_R if w_F = w_L

beta_dd = 2/(3 * (k_L**2) * ((w_L * 10**(-6))**2)) # Calculation of beta_dd from grid parameters

# Creating the spatial domain

L = (w_L * N_w) * (1 / xy_scaling) # Total grid length

x = numpy.linspace(-L/2,L/2,Nx) # x-vector setup
y = numpy.linspace(-L/2,L/2,Ny) # y-vector setup
dx = x[1] - x[0]
dy = y[1] - y[0]
[X, Y] = numpy.meshgrid(x,y) # Creating 2D meshgrid

x_physical = (x * xy_scaling) # microns
y_physical = (y * xy_scaling) # microns

# Creating the k-space far field domain

n = numpy.arange(-Nx/2, Nx/2, 1).tolist()
k_wavenum = numpy.array(n, dtype=int)*((2 * numpy.pi) / L ) # k-space grid creation
dk = k_wavenum[1] - k_wavenum[0]
[Ex, Ey] = numpy.meshgrid(k_wavenum,k_wavenum) # k-space meshgrid for far-field treatments

# Optical and related z-axis conversions

N_opt = w_F / w_L # Ratio of optical profile mode to scaling waist: ideally, this equals one
expected_z_R_position = 0.5 * (N_opt**2) # Where one would expect to find z_R, in computational units
expected_z_R = (0.5 * k_L * (w_F / 10**6)**2) * 10**3 # Physical units of profile z_R, in mm
dt = dx**2 / 4 # step size in tau, ensures anti-aliasing satisfaction

###############################################################################
# Initial spatial profile creation
###############################################################################

# Atoms - Thomas Fermi distribution

Psi = CACFuncs.TFDist(X, Y, BEC_w_F, w_L, Psi_amp, noise_amplitude)
max_Psi = numpy.max(numpy.abs(Psi))

# Pump - LG mode

A_in = CACFuncs.LGMode(X, Y, N_opt, l_mode, l_mode_prof, p_mode, OAM, F_amp, noise_amplitude)

# Intra-cavity field - noise

A = CACFuncs.Noisy(X, Y, F_amp, noise_amplitude)
max_A = numpy.max(numpy.abs(A))

# Absorbing boundary creation

boundaries = CACFuncs.Boundaries(x, boundary_extent, boundary_grad, N_w, X, Y)

###############################################################################
# Final pre-loop variable setup and initial field storage
###############################################################################

current_tau = 0 # Current tau-value, updated throughout loop
i = 0 # Number of iterations around loop
target_data_tau = data_fac # First place to extract and save field data
target_plot_tau = plot_fac # First place to extract and plot fields
break_command = 0 # Command to break loop if numerical instability is reached

# Atomic parameter adaptations to account for model prefactor
beta_col_pre = beta_col
beta_col = beta_col * atomic_prefac
L_3_pre = L_3
L_3 = L_3_pre * atomic_prefac
s_atoms = s * atomic_prefac
beta_dd_atoms = beta_dd * atomic_prefac

# Configuring data and plot outputs, and outputting initial fields
if data_fac != 0 and plot_fac != 0 :
    data_directory, plot_directory = CACFuncs.DirSetup(data_fac, plot_fac, output_directory, overwrite) # Setting up the directories
    data_int = 0 # Integer index for data storage
    output_fields_data = [Psi, A, A_in] # Fields you wish to output as data
    data_int = CACFuncs.DataOutput(data_int, output_fields_data, data_directory) # Output the data
    plot_int = 0 # Integer index for plotting
    output_fields_plot = [Psi, A, A_in] # Fields you wish to output as plots
    tracked_maxs = numpy.zeros(len(output_fields_plot)) # Setup for tracked maxs (irregardless of on or off)
    plot_int, tracked_maxs = CACFuncs.PlotOutput(plot_int, output_fields_plot, track_to_max, tracked_maxs, x_physical, y_physical, plot_directory) # Output the plots
elif data_fac != 0 :
    data_directory, plot_directory = CACFuncs.DirSetup(data_fac, plot_fac, output_directory, overwrite) # Setting up the directories
    data_int = 0 # Integer index for data storage
    output_fields_data = [Psi, A, A_in] # Fields you wish to output as data
    data_int = CACFuncs.DataOutput(data_int, output_fields_data, data_directory) # Output the data
elif plot_fac != 0 :
    data_directory, plot_directory = CACFuncs.DirSetup(data_fac, plot_fac, output_directory, overwrite) # Setting up the directories
    plot_int = 0 # Integer index for plotting
    output_fields_plot = [Psi, A, A_in] # Fields you wish to output as plots
    tracked_maxs = numpy.zeros(len(output_fields_plot)) # Setup for tracked maxs (irregardless of on or off)
    plot_int, tracked_maxs = CACFuncs.PlotOutput(plot_int, output_fields_plot, track_to_max, tracked_maxs, x_physical, y_physical, plot_directory) # Output the plots
else :
    print('You\'re not outputting anything from this job...')
    sys.exit()

# Record run info to data file
grid_params = [final_tau, dt, w_L, N_w, L, Nx, dx, x, dk, k_wavenum, boundary_extent] # Grid parameters for recording
optical_params = [w_F, F_amp, l_mode, l_mode_prof, OAM, p_mode, F_amp, A] # Initial optical parameters for recording
atomic_params = [BEC_w_F, Psi_amp, Psi] # Initial atomic parameters for recording
nonlinearities = ['s', s, 'alpha/kappa', atomic_prefac, 'beta_dd', beta_dd, 'beta_col', beta_col_pre, 'L_3', L_3_pre, 'theta', theta, 'alpha_F', alpha_light, 'beta_F', beta_light, 'sigma', sigma] # Nonlinear term names and strengths for recording
output_configs = [data_fac, plot_fac, track_to_max]
CACFuncs.InfoFileCommence(__file__, output_directory, grid_params, optical_params, atomic_params, nonlinearities, output_configs) # Record to data file
    
###############################################################################
# Split-step spatial evolution loop
###############################################################################

# Initial move to k-space
Psi_fft = numpy.fft.fftshift(numpy.fft.fft2(Psi))/(Nx*Ny) 
A_fft = numpy.fft.fftshift(numpy.fft.fft2(A))/(Nx*Ny)

while current_tau <= final_tau and break_command == 0 : 
    
    # Half-step application in k-space
    Psi_fft = numpy.exp(-1j * atomic_prefac * ((Ex**2 + Ey**2) * dt / 2)) * Psi_fft 
    A_fft = numpy.exp(-1j * (( (1 / 1j) - theta + (alpha_light*(Ex**2 + Ey**2))) * dt / 2)) * A_fft
    
    # Return to real space
    Psi = numpy.fft.ifft2(numpy.fft.fftshift(Psi_fft))*(Nx*Ny)
    A = numpy.fft.ifft2(numpy.fft.fftshift(A_fft))*(Nx*Ny)
    
    # RK application of full space step in real space
    Psi_h = Psi
    A_h = A
    dPsidt = (1j * - ((s_atoms*numpy.abs(A_h)**2) - (beta_dd_atoms * numpy.abs(A_h)**2 * numpy.abs(Psi_h)**2) + (beta_col * numpy.abs(Psi_h)**2) - (1j * L_3 * numpy.abs(Psi_h)**4))) * Psi_h
    dAdt = (1j * beta_light * ((- s*numpy.absolute(Psi_h)**2 + beta_dd*numpy.absolute(Psi_h)**4))/(1 + sigma*numpy.abs(A_h)**2)) * A_h
    dAdt = A_in + dAdt
    Psi_t = Psi_h + dt*dPsidt/2
    A_t = A_h + dt*dAdt/2
    dPsidt = (1j * - ((s_atoms*numpy.abs(A_t)**2) - (beta_dd_atoms * numpy.abs(A_t)**2 * numpy.abs(Psi_t)**2) + (beta_col * numpy.abs(Psi_t)**2) - (1j * L_3 * numpy.abs(Psi_t)**4))) * Psi_t
    dAdt = (1j * beta_light * ((- s*numpy.abs(Psi_t)**2 + beta_dd*numpy.abs(Psi_t)**4))/(1 + sigma*numpy.abs(A_t)**2)) * A_t
    dAdt = A_in + dAdt
    Psi = Psi_h + dt*dPsidt
    A = A_h + dt*dAdt
    
    # Boundary application in real space
    Psi = Psi * boundaries
    A = A * boundaries
        
    # Return to k-space
    Psi_fft = numpy.fft.fftshift(numpy.fft.fft2(Psi))/(Nx*Ny)
    A_fft = numpy.fft.fftshift(numpy.fft.fft2(A))/(Nx*Ny)
    
    # Half-step application in k-space
    Psi_fft = numpy.exp(-1j * atomic_prefac * ((Ex**2 + Ey**2) * dt / 2)) * Psi_fft 
    A_fft = numpy.exp(-1j * (( (1 / 1j) - theta + (alpha_light*(Ex**2 + Ey**2))) * dt / 2)) * A_fft
    
    current_tau += dt
    i += 1
    
    # End of loop procedure, now for checks and data processing
    
    break_condition = float(numpy.max(numpy.abs(Psi))) # If breakdown has occured in atomic field
    if numpy.isnan(break_condition) == True :
        break_command = 1
        
    if current_tau >= target_data_tau and data_fac != 0 :
        # Return fields to real space
        Psi = numpy.fft.ifft2(numpy.fft.fftshift(Psi_fft))*(Nx*Ny)
        A = numpy.fft.ifft2(numpy.fft.fftshift(A_fft))*(Nx*Ny)
        output_fields_data = [Psi, A] # Fields you wish to output as data
        data_int = CACFuncs.DataOutput(data_int, output_fields_data, data_directory) # Output the data
        target_data_tau += data_fac
        
    if current_tau >= target_plot_tau and plot_fac != 0 :
        # Return fields to real space
        Psi = numpy.fft.ifft2(numpy.fft.fftshift(Psi_fft))*(Nx*Ny)
        A = numpy.fft.ifft2(numpy.fft.fftshift(A_fft))*(Nx*Ny)
        output_fields_plot = [Psi, A, A_in] # Fields you wish to output as plots
        plot_int, tracked_maxs = CACFuncs.PlotOutput(plot_int, output_fields_plot, track_to_max, tracked_maxs, x_physical, y_physical, plot_directory) # Output the plots
        target_plot_tau += plot_fac
        
###############################################################################
# Post-loop storage and plotting
###############################################################################

if data_fac != 0 :
    # Return fields to real space
    Psi = numpy.fft.ifft2(numpy.fft.fftshift(Psi_fft))*(Nx*Ny)
    A = numpy.fft.ifft2(numpy.fft.fftshift(A_fft))*(Nx*Ny)
    data_int = 'Fin'
    output_fields_data = [Psi, A] # Fields you wish to output as data
    data_int = CACFuncs.DataOutput(data_int, output_fields_data, data_directory) # Output the data
    
if plot_fac != 0 :
    # Return fields to real space
    Psi = numpy.fft.ifft2(numpy.fft.fftshift(Psi_fft))*(Nx*Ny)
    A = numpy.fft.ifft2(numpy.fft.fftshift(A_fft))*(Nx*Ny)
    plot_int = 'Fin'
    output_fields_plot = [Psi, A, A_in] # Fields you wish to output as plots
    plot_int, tracked_maxs = CACFuncs.PlotOutput(plot_int, output_fields_plot, track_to_max, tracked_maxs, x_physical, y_physical, plot_directory) # Output the plots
    
# Record final info to data file
CACFuncs.InfoFileTerminate(__file__, output_directory, current_tau) # Record to data file