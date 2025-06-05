# Grant W. Henderson, University of Strathclyde
# grant.henderson@strath.ac.uk
# Contains functions written for the FOAMily (and friends)
# They are used in various 2D SSFM codes for light / atom / atom-light (co-)propagation and cavity systems
# Last update: 29/01/2025

###############################################################################
# Packages
###############################################################################

import numpy
from scipy.special import assoc_laguerre
import time
import os
import matplotlib.pyplot as plt
from datetime import date, datetime
import scipy

###############################################################################
# Field definitions
###############################################################################

# Creates a Laguerre-Gaussian mode at the beam waist
def LGMode(X, Y, N_opt, l_mode, l_mode_prof, p_mode, OAM, F_amp, noise_amplitude) :
    w_0_comp = N_opt # w_0 in terms of scaled waist
    w_0_sq_comp = w_0_comp**2 # (w_0)**2 in terms of scaled waist
    gaus = (X**2  + Y**2) / (w_0_sq_comp) # Gaussian profile in scaled units
    Phi = l_mode_prof * numpy.arctan2(Y,X) # Argument of exponent in LG mode
    Term1 = (numpy.sqrt(gaus))**numpy.absolute(l_mode)
    Term2 = assoc_laguerre(2*gaus, p_mode, numpy.absolute(l_mode))
    Term3 = numpy.exp(-gaus/2)
    if OAM == 1 and isinstance(Y, int) == False :
        Term4 = numpy.exp(1j*Phi)
    else :
        Term4 = 1
    A = Term1 * Term2 * Term3 * Term4 # Creation of LG beam
    max_A = numpy.max(numpy.abs(A))
    A = F_amp * (A / max_A) # Re-scaled for desired maximum amplitude F_amp
    if isinstance(Y, int) == False :
        noise_re = numpy.random.uniform(-1,1,(len(X),len(Y)))
        noise_im = numpy.random.uniform(-1,1,(len(X),len(Y)))
    else :
        noise_re = numpy.random.uniform(-1,1,(len(X)))
        noise_im = numpy.random.uniform(-1,1,(len(X)))
    A = A + (F_amp*noise_amplitude*noise_re) + 1j*(F_amp*noise_amplitude*noise_im) # Applying noise to field
    return A

def BGMode(X, Y, w_F, w_L, l_mode, OAM, gaus_width, kappa, amp, noise_amplitude) :
    w_0_comp = w_F / w_L # w_0 in terms of scaled waist
    w_0_sq_comp = w_0_comp**2 # (w_0)**2 in terms of scaled waist
    t = (X**2  + Y**2) / (w_0_sq_comp)
    Term1 = scipy.special.spherical_jn(int(numpy.abs(l_mode)), kappa*numpy.sqrt(t))
    Term1 = Term1 / (numpy.max(Term1))
    Gaus = numpy.exp(-((X**2  + Y**2) / (w_0_sq_comp * gaus_width**2)))
    if OAM == 1 and isinstance(Y, int) == False :
        Phi = l_mode * numpy.arctan2(Y,X)
        Term4 = numpy.exp(1j*Phi)
    else :
        Term4 = 1
    field = Term1 * Term4 * Gaus
    max_field = numpy.max(numpy.abs(field))
    field = amp * (field / max_field)
    if isinstance(Y, int) == False :
        noise_re = numpy.random.uniform(-1,1,(len(X),len(Y)))
        noise_im = numpy.random.uniform(-1,1,(len(X),len(Y)))
    else :
        noise_re = numpy.random.uniform(-1,1,(len(X)))
        noise_im = numpy.random.uniform(-1,1,(len(X)))
    field = field + (amp*noise_amplitude*noise_re) + 1j*(amp*noise_amplitude*noise_im) # Applying noise to field
    return field

# Creates a field of noise
def Noisy(X, Y, amp, noise_amplitude) :
    if isinstance(Y, int) == False :
        noise_re = numpy.random.uniform(-1,1,(len(X),len(Y)))
        noise_im = numpy.random.uniform(-1,1,(len(X),len(Y)))
    else :
        noise_re = numpy.random.uniform(-1,1,(len(X)))
        noise_im = numpy.random.uniform(-1,1,(len(X)))
    A = (amp*noise_amplitude*noise_re) + 1j*(amp*noise_amplitude*noise_im) # Applying noise to field
    return A

# Creates a Thomas-Fermi distribution
def TFDist(X, Y, BEC_w_F, w_L, Psi_amp, noise_amplitude) :
    N_BEC = BEC_w_F / w_L # Beam waist relative to computational grid
    w_0_BEC_comp = N_BEC
    w_0_sq_BEC_comp = w_0_BEC_comp**2

    Term1 = (X**2 / (2 * w_0_sq_BEC_comp))
    Term2 = (Y**2 / (2 * w_0_sq_BEC_comp)) 
    Psi = Psi_amp * (1 - Term1 - Term2) # Thomas Fermi distribution, max. amp. Psi_amp
    Psi = Psi.clip(min=0) # Removing terms < 0
    if isinstance(Y, int) == False :
        noise_re = numpy.random.uniform(-1,1,(len(X),len(Y)))
        noise_im = numpy.random.uniform(-1,1,(len(X),len(Y)))
    else :
        noise_re = numpy.random.uniform(-1,1,(len(X)))
        noise_im = numpy.random.uniform(-1,1,(len(X)))
    Psi = Psi + (Psi_amp*noise_amplitude*noise_re) + 1j*(Psi_amp*noise_amplitude*noise_im) # Applying noise to field
    return Psi

# Creates a profile used for absorbing boundaries
def Boundaries(x, boundary_extent, boundary_grad, N_w, X, Y) :
    boundary_lim = numpy.max(x) * boundary_extent # Position of the absorbing boundaries relative to extent of grid 
    boundaries = 1-numpy.tanh((boundary_grad/(N_w / 30))*(numpy.sqrt(X**2 + Y**2) - boundary_lim))
    boundaries = numpy.interp(boundaries, (boundaries.min(), boundaries.max()), (0, 1)) # Re-scaled between 0 and 1
    return boundaries

# Creates a homogeneous (planar) field
def Homogeneous(X, Y, amp, noise_amplitude) :
    if isinstance(Y, int) == False :
        noise_re = numpy.random.uniform(-1,1,(len(X),len(Y)))
        noise_im = numpy.random.uniform(-1,1,(len(X),len(Y)))
    else :
        noise_re = numpy.random.uniform(-1,1,(len(X)))
        noise_im = numpy.random.uniform(-1,1,(len(X)))
    A = amp + (amp*noise_amplitude*noise_re) + 1j*(amp*noise_amplitude*noise_im) # Applying noise to field
    return A

# Creates a top hat with the capacity for OAM application
def TopHat(X, Y, amp, x, w_F, N_w, w_L, l_mode, grad, noise_amplitude) :
    w_F_scale = numpy.max(x) * (w_F / (0.5 * N_w * w_L))
    field = amp * (0.5 * (1 - numpy.tanh(grad*(numpy.sqrt(X**2 + Y**2) - w_F_scale))))
    if isinstance(Y, int) == False :
        Phi = l_mode * numpy.arctan2(Y,X)
        field = field * numpy.exp(1j * Phi)
        noise_re = numpy.random.uniform(-1,1,(len(X),len(Y)))
        noise_im = numpy.random.uniform(-1,1,(len(X),len(Y)))
    else :
        noise_re = numpy.random.uniform(-1,1,(len(X)))
        noise_im = numpy.random.uniform(-1,1,(len(X)))
    field = field + (amp*noise_amplitude*noise_re) + 1j*(amp*noise_amplitude*noise_im) # Applying noise to field
    return field

###############################################################################
# Data / plot outputs
###############################################################################

# Sets up run result directories according to desired outputs
def DirSetup(data_fac, plot_fac, output_directory, overwrite) :
    if data_fac != 0 :
        data_directory = './'+str(output_directory)+'/Data/'
        directory_name = os.path.join(str(data_directory))
        if overwrite == 0 :
            os.makedirs(directory_name)
        else :
            try :
                os.makedirs(directory_name)
            except FileExistsError :
                time.sleep(10**-5) # Avoids termination error
    else :
        data_directory = ''
    if plot_fac != 0 :
        plot_directory = './'+str(output_directory)+'/Plot/'
        directory_name = os.path.join(str(plot_directory))
        if overwrite == 0 :
            os.makedirs(directory_name)
        else :
            try :
                os.makedirs(directory_name)
            except FileExistsError :
                time.sleep(10**-5) # Avoids termination error
    else :
        plot_directory = ''
    return data_directory, plot_directory
                
def DataOutput(data_int, output_fields_data, data_directory) :
    field_int = 1
    for curr_field in output_fields_data :
        fld_re = numpy.real(curr_field) # Real part of field
        fld_im = numpy.imag(curr_field) # Imaginary part of field
        numpy.savetxt(str(data_directory)+"/Field"+str(int(field_int))+"Re_"+str(data_int)+".csv", fld_re, delimiter=",")
        numpy.savetxt(str(data_directory)+"/Field"+str(int(field_int))+"Im_"+str(data_int)+".csv", fld_im, delimiter=",")
        field_int += 1
    if isinstance(data_int, str) == False :
        data_int += 1
    return data_int
        
def PlotOutput(plot_int, output_fields_plot, track_to_max, tracked_maxs, x_physical, y_physical, plot_directory) :
    rows = len(output_fields_plot)
    # Setup a plot of however many fields we want to plot in total (amplitude and phase of each)
    fig, ax = plt.subplots(2,int(rows))
    # Loop through however many fields we are plotting
    field_int = 0
    for curr_field, curr_max in zip(output_fields_plot, tracked_maxs) :
        # Setup tracking to maximum amplitudes in time, if desired
        if track_to_max == 1 :
            if numpy.max(numpy.abs(curr_field)) > tracked_maxs[field_int] :
                tracked_maxs[field_int] = numpy.max(numpy.abs(curr_field))
            vmax_curr = tracked_maxs[field_int]
        else :
            vmax_curr = numpy.max(numpy.abs(curr_field))
        if rows == 1 :
            if isinstance(y_physical, int) == False :
                ax[0].imshow(numpy.abs(curr_field), vmin=0, vmax=vmax_curr, extent=[x_physical.min(), x_physical.max(), y_physical.min(), y_physical.max()])
                ax[0].axis('off')
                ax[1].imshow(numpy.angle(curr_field), cmap=plt.cm.hsv, vmin=-numpy.pi, vmax=numpy.pi, extent=[x_physical.min(), x_physical.max(), y_physical.min(), y_physical.max()])    
                ax[1].axis('off')
            else :
                ax[0].plot(x_physical, numpy.abs(curr_field), c='k', linewidth=1)
                ax[0].set_ylim(bottom=0, top=vmax_curr)
                ax[0].set_xlim(left=x_physical.min(), right=x_physical.max())
                ax[0].tick_params(axis='both', which='major', labelsize=3)
                ax[1].plot(x_physical, numpy.angle(curr_field), c='k', linewidth=1)
                ax[1].set_ylim(bottom=-numpy.pi, top=numpy.pi)
                ax[1].set_xlim(left=x_physical.min(), right=x_physical.max())
                ax[1].tick_params(axis='both', which='major', labelsize=3)
        else :
            if isinstance(y_physical, int) == False :
                ax[0, field_int].imshow(numpy.abs(curr_field), vmin=0, vmax=vmax_curr, extent=[x_physical.min(), x_physical.max(), y_physical.min(), y_physical.max()])
                ax[0, field_int].axis('off')
                ax[1, field_int].imshow(numpy.angle(curr_field), cmap=plt.cm.hsv, vmin=-numpy.pi, vmax=numpy.pi, extent=[x_physical.min(), x_physical.max(), y_physical.min(), y_physical.max()])    
                ax[1, field_int].axis('off')
            else :
                ax[0, field_int].plot(x_physical, numpy.abs(curr_field), c='k', linewidth=1)
                ax[0, field_int].set_ylim(bottom=0, top=vmax_curr)
                ax[0, field_int].set_xlim(left=x_physical.min(), right=x_physical.max())
                ax[0, field_int].tick_params(axis='both', which='major', labelsize=3)
                ax[1, field_int].plot(x_physical, numpy.angle(curr_field), c='k', linewidth=1)
                ax[1, field_int].set_ylim(bottom=-numpy.pi, top=numpy.pi)
                ax[1, field_int].set_xlim(left=x_physical.min(), right=x_physical.max())
                ax[1, field_int].tick_params(axis='both', which='major', labelsize=3)
        field_int += 1
    plt.savefig(str(plot_directory)+'/Fields_'+str(plot_int)+'.png', bbox_inches='tight', dpi=200)
    plt.close()
    if isinstance(plot_int, str) == False :
        plot_int += 1
    return plot_int, tracked_maxs

def InfoFileCommence(script, output_directory, grid_params, optical_params, atomic_params, nonlinearities, output_configs) :
    f = open(str(output_directory)+'/Info.txt',"w+")
    f.write("This is the readme for the most recent run of this simulation. \n")
    f.write("Script: "+str(script)+" \n")
    today = date.today()
    now = datetime.now()
    f.write("Commenced on: "+str(today.strftime("%d/%m/%Y"))+', '+str(now.strftime("%H:%M:%S"))+" \n \n")
    f.write("########## NUMERICAL PARAMETERS ##########\n")
    if 'propagation' in script or 'evolution' in script :
        f.write("zeta (z) = "+str(grid_params[0])+" \n")
        if grid_params[1] == 'dynamic' :
            f.write("dz dynamically calculated \n \n")
        else :
            f.write("dz = "+str(grid_params[1])+" \n \n")
    elif 'cavity' in script :
        f.write("tau (t) = "+str(grid_params[0])+" \n")
        if grid_params[1] == 'dynamic' :
            f.write("dt dynamically calculated \n \n")
        else :
            f.write("dt = "+str(grid_params[1])+" \n \n")
    f.write("w_L = "+str(grid_params[2])+" \n")
    f.write("N_w = "+str(grid_params[3])+" \n")
    f.write("L = "+str(grid_params[4])+" \n")
    f.write("Nx = Ny = "+str(grid_params[5])+"\n")
    f.write("dx = dy = "+str(grid_params[6])+"\n")
    f.write("max(x) = max(y) = "+str(numpy.max(grid_params[7]))+"\n")
    f.write("dk = "+str(grid_params[8])+" \n")
    f.write("max(k) = "+str(numpy.max(grid_params[9]))+"\n")
    f.write("boundary limits = "+str(grid_params[10])+" \n \n")
    if grid_params[1] != 'dynamic' :
        aa_param = (grid_params[6]**2) / (numpy.pi * grid_params[1])
        f.write("(dx^2) / (pi * dt) = "+str(aa_param)+" \n \n")
    if optical_params != 0 :
        if 'propagation' in script :
            f.write("########## OPTICAL PARAMETERS ##########\n")
        elif 'cavity' in script :
            f.write("########## PUMP PARAMETERS ##########\n")
        elif 'example' in script :
            f.write("########## FIELD PARAMETERS ##########\n")
        f.write("w_F = "+str(optical_params[0])+" \n")
        f.write("A_F = "+str(optical_params[1])+" \n")
        f.write("ell = "+str(optical_params[2])+" \n")
        f.write("profile_ell = "+str(optical_params[3])+" \n")
        f.write("OAM = "+str(optical_params[4])+" \n")
        f.write("p = "+str(optical_params[5])+" \n")
        f.write("F_amp = "+str(optical_params[6])+" \n")
        f.write("max(F) = "+str(numpy.max(numpy.abs(optical_params[7])))+"\n \n")
    if atomic_params != 0 :
        f.write("########## ATOMIC PARAMETERS ##########\n")
        f.write("w_BEC = "+str(atomic_params[0])+" \n")
        f.write("Psi_amp = "+str(atomic_params[1])+" \n")
        f.write("max(Psi) = "+str(numpy.max(numpy.abs(atomic_params[2])))+"\n \n")
    if nonlinearities != 0 :
        f.write("########## NONLINEARITIES ##########\n")
        for term_name, term_size in zip(nonlinearities[0::2], nonlinearities[1::2]) :
            f.write(str(term_name)+" = "+str(term_size)+" \n")
        f.write("\n")
    f.write("########## OUTPUT CONFIG ##########\n")
    f.write("data_fac = "+str(output_configs[0])+" \n")
    f.write("plot_fac = "+str(output_configs[1])+" \n")
    f.write("track_to_max = "+str(output_configs[2])+" \n \n")
    f.close()
    
def InfoFileTerminate(script, output_directory, current) :
    f = open(str(output_directory)+'/Info.txt',"a+")
    f.write("########## POST-RUN INFO ##########\n")
    today = date.today()
    now = datetime.now()
    f.write("Finished on: "+str(today.strftime("%d/%m/%Y"))+', '+str(now.strftime("%H:%M:%S"))+" \n")
    if 'propagation' in script :
        f.write("z = "+str(current))
    elif 'cavity' in script :
        f.write("t = "+str(current))
    f.close()
    
###############################################################################
# Elimination Methods
###############################################################################

# Sets up finite grid and solves for an intra-cavity field
def Elim_FiniteGrid(Nx, Ny, s, atoms, beta_dd, A_in, theta, alpha_light, beta_light, dx) :
    nlt = (-(s*numpy.abs(atoms)**2) + (beta_dd*numpy.abs(atoms)**4))
    if Ny != 0 :
        term_matrix = 1j*numpy.zeros((Ny, Nx, Ny, Nx))
        y_index = 0
        while y_index < Nx :
            x_index = 0
            while x_index < Nx :
                term_matrix[y_index, (x_index-1)%Nx, y_index, x_index] += (-1j * alpha_light) / (dx**2)
                term_matrix[(y_index-1)%Nx, x_index, y_index, x_index] += (-1j * alpha_light) / (dx**2)
                term_matrix[y_index, x_index, y_index, x_index] += 1 - (1j*theta) + ((4j * alpha_light) / (dx**2)) - (1j * beta_light * nlt[y_index, x_index])
                term_matrix[y_index, (x_index+1)%Nx, y_index, x_index] += (-1j * alpha_light) / (dx**2)
                term_matrix[(y_index+1)%Nx, x_index, y_index, x_index] += (-1j * alpha_light) / (dx**2)
                x_index += 1
            y_index += 1
        field_prog = numpy.linalg.tensorsolve(term_matrix, A_in)
    else :
        term_matrix = 1j*numpy.zeros((Nx,Nx))
        x_index = 0
        while x_index < Nx :
            term_matrix[(x_index-1)%Nx, x_index] += (-1j * alpha_light) / (dx**2)
            term_matrix[x_index, x_index] += 1 - (1j*theta) + ((2j * alpha_light) / (dx**2)) - (1j * beta_light * nlt[x_index])
            term_matrix[(x_index+1)%Nx, x_index] += (-1j * alpha_light) / (dx**2)
            x_index += 1
        field_prog = numpy.linalg.solve(term_matrix, A_in)
    return field_prog

# Evolves a field to defined steady state criterion, used for 'relaxation' approach
def Elim_Relax(field, Psi, A_in, boundaries, Nx, Ny, Ex, Ey, dt, theta, alpha_light, beta_light, beta_dd, s, sigma) :
    field_fft = numpy.fft.fftshift(numpy.fft.fft2(field))/(Nx*Ny)
    field_track = numpy.random.rand(Nx, Ny, 10) + 1j*numpy.random.rand(Nx, Ny, 10)
    while numpy.max(numpy.var(numpy.abs(field_track),2)) >= 10**(-5) :
        field_fft = numpy.exp(-1j * (( (1 / 1j) - theta + (alpha_light*(Ex**2 + Ey**2))) * dt / 2)) * field_fft
        field = numpy.fft.ifft2(numpy.fft.fftshift(field_fft))*(Nx*Ny)
        
        field_h = field
        dfielddt = (1j * - beta_light * ((s*numpy.abs(Psi)**2 - beta_dd*numpy.abs(Psi)**4))/(1 + sigma*numpy.abs(field_h)**2)) * field_h
        dfielddt = A_in + dfielddt
        field_t = field_h + dt*dfielddt/2
        dfielddt = (1j * - beta_light * ((s*numpy.abs(Psi)**2 - beta_dd*numpy.abs(Psi)**4))/(1 + sigma*numpy.abs(field_h)**2)) * field_t
        dfielddt = A_in + dfielddt
        field = field_h + dt*dfielddt
        
        field = field * boundaries
        field_track[:,:,0:-1] = field_track[:,:,1:]
        field_track[:,:,-1] = field
        
        field_fft = numpy.fft.fftshift(numpy.fft.fft2(field))/(Nx*Ny)
        field_fft = numpy.exp(-1j * (( (1 / 1j) - theta + (alpha_light*(Ex**2 + Ey**2))) * dt / 2)) * field_fft
    field = numpy.fft.ifft2(numpy.fft.fftshift(field_fft))*(Nx*Ny)
    return field

# Evolves a field to defined steady state criterion, used for 'relaxation' approach, implementing a Runge-Kutta-Fehlberg 45 application with dynamic time step
def Elim_Relax45(field, Psi, A_in, boundaries, Nx, Ny, Ex, Ey, dt, theta, alpha_light, beta_light, beta_dd, s, sigma, rkf_threshold) :
    field_fft = numpy.fft.fftshift(numpy.fft.fft2(field))/(Nx*Ny)
    field_track = numpy.random.rand(Nx, Ny, 10) + 1j*numpy.random.rand(Nx, Ny, 10)
    while numpy.max(numpy.var(numpy.abs(field_track),2)) >= 10**(-5) :
        err_est_field = numpy.ones((Nx, Ny))
        while numpy.max(err_est_field) > rkf_threshold :
            field_fft = numpy.exp(-1j * (( (1 / 1j) - theta + (alpha_light*(Ex**2 + Ey**2))) * dt / 2)) * field_fft
            field = numpy.fft.ifft2(numpy.fft.fftshift(field_fft))*(Nx*Ny)
            
            # field_h = field
            # dfielddt = (1j * - beta_light * ((s*numpy.abs(Psi)**2 - beta_dd*numpy.abs(Psi)**4))/(1 + sigma*numpy.abs(field_h)**2)) * field_h
            # dfielddt = A_in + dfielddt
            # field_t = field_h + dt*dfielddt/2
            # dfielddt = (1j * - beta_light * ((s*numpy.abs(Psi)**2 - beta_dd*numpy.abs(Psi)**4))/(1 + sigma*numpy.abs(field_h)**2)) * field_t
            # dfielddt = A_in + dfielddt
            # field = field_h + dt*dfielddt
            
            # Runge-Kutta-Fehlberg 45 application of full space step in real space
            field_0 = field
            dfielddt_1 = (1j * - beta_light * ((s*numpy.abs(Psi)**2 - beta_dd*numpy.abs(Psi)**4))/(1 + sigma*numpy.abs(field_0)**2)) * field_0
            dfielddt_1 = A_in + dfielddt_1
            field_1 = field_0 + ((1)*dt*dfielddt_1)
            dfielddt_2 = (1j * - beta_light * ((s*numpy.abs(Psi)**2 - beta_dd*numpy.abs(Psi)**4))/(1 + sigma*numpy.abs(field_1)**2)) * field_1
            dfielddt_2 = A_in + dfielddt_2
            field_2 = field_0 + ((1/4)*dt*dfielddt_2)
            dfielddt_3 = (1j * - beta_light * ((s*numpy.abs(Psi)**2 - beta_dd*numpy.abs(Psi)**4))/(1 + sigma*numpy.abs(field_2)**2)) * field_2
            dfielddt_3 = A_in + dfielddt_3
            field_3 = field_0 + ((3/32)*dt*dfielddt_2) + ((9/32)*dt*dfielddt_3)
            dfielddt_4 = (1j * - beta_light * ((s*numpy.abs(Psi)**2 - beta_dd*numpy.abs(Psi)**4))/(1 + sigma*numpy.abs(field_3)**2)) * field_3
            dfielddt_4 = A_in + dfielddt_4
            field_4 = field_0 + ((1932/2197)*dt*dfielddt_2) + ((- 7200/2197)*dt*dfielddt_3) + ((7296/2197)*dt*dfielddt_4)
            dfielddt_5 = (1j * - beta_light * ((s*numpy.abs(Psi)**2 - beta_dd*numpy.abs(Psi)**4))/(1 + sigma*numpy.abs(field_4)**2)) * field_4
            dfielddt_5 = A_in + dfielddt_5
            field_5 = field_0 + ((439/216)*dt*dfielddt_2) + ((- 8)*dt*dfielddt_3) + ((3680/513)*dt*dfielddt_4) + ((- 845/4104)*dt*dfielddt_5)
            dfielddt_6 = (1j * - beta_light * ((s*numpy.abs(Psi)**2 - beta_dd*numpy.abs(Psi)**4))/(1 + sigma*numpy.abs(field_5)**2)) * field_5
            dfielddt_6 = A_in + dfielddt_6
            field_6 = field_0 + ((- 8/27)*dt*dfielddt_2) + ((2)*dt*dfielddt_3) + ((- 3544/2565)*dt*dfielddt_4) + ((1859/4104)*dt*dfielddt_5) + ((- 11/40)*dt*dfielddt_6)
                
            field_RKF4 = field_0 + ((25/216)*dt*dfielddt_1) + ((0)*dt*dfielddt_2) + ((1408/2565)*dt*dfielddt_3) + ((2197/4104)*dt*dfielddt_4) + ((- 1/5)*dt*dfielddt_5) + ((0)*dt*dfielddt_6)
            
            field_RKF5 = field_0 + ((16/135)*dt*dfielddt_1) + ((0)*dt*dfielddt_2) + ((6656/12825)*dt*dfielddt_3) + ((28561/56430)*dt*dfielddt_4) + ((- 9/50)*dt*dfielddt_5) + ((2/55)*dt*dfielddt_6)
            
            # Estimation of error between RKF4 and RKF5 solutions
            err_est_field = numpy.abs((25/216 - 16/216)*dt*dfielddt_1) + numpy.abs((0 - 0)*dt*dfielddt_2) + numpy.abs((1408/2565 - 6656/12825)*dt*dfielddt_3) + numpy.abs((2197/4104 - 28561/56430)*dt*dfielddt_4) + numpy.abs((- 1/5 - 9/50)*dt*dfielddt_5) + numpy.abs((0 - 2/55)*dt*dfielddt_6)
            
            # Re-calculation of step size based on error estimation, and acceptance / rejection of step
            if numpy.max(err_est_field) > rkf_threshold :
                dt = 0.9 * dt * (rkf_threshold / numpy.max(err_est_field))**(1/5)
            else :
                dt_next = 0.9 * dt * (rkf_threshold / numpy.max(err_est_field))**(1/5)
        
        # Once error criterion is met, continue with the most accurate solutions of the above (i.e. the RKF 5th order result)
        field = field_RKF5

        field = field * boundaries
        field_track[:,:,0:-1] = field_track[:,:,1:]
        field_track[:,:,-1] = field
        
        field_fft = numpy.fft.fftshift(numpy.fft.fft2(field))/(Nx*Ny)
        field_fft = numpy.exp(-1j * (( (1 / 1j) - theta + (alpha_light*(Ex**2 + Ey**2))) * dt / 2)) * field_fft
        dt = dt_next
    field = numpy.fft.ifft2(numpy.fft.fftshift(field_fft))*(Nx*Ny)
    return field