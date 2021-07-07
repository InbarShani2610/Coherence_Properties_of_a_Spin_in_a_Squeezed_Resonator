
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from matplotlib import interactive
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from pathlib import Path
import math
from scipy.optimize import curve_fit

## returns file name and directory 

def Save_func(folder_name,t_i,t_f,resolution,N,omega_r,omega_q,kappa,g,r,gamma):

    filename = \
        '''taui{0:e}
        tauf{1:e}
        resolution{2:e}
        N{3:e}
        omega_r_tilde{4:e}
        omega_q_tilde{5:e}
        kappa{6:e}
        g{7:e}
        r{8:e}
        gamma{9:e}
        .npz
        '''.format(t_i,t_f,resolution,N,omega_r,omega_q,kappa,g,r,gamma).replace('\n','').replace(' ','')

    save_loc_dir = Path.cwd() / folder_name

    if not save_loc_dir.exists():
        save_loc_dir.mkdir()
    save_loc = save_loc_dir / filename
    
    return save_loc

## saves and returns spectrum calculations

def Calc_spec(save_loc,H_m,wlist,c_ops,a,r,RECALC):
    spec_max_r = None
    res = None
    Width_vec_i = None

    if (not RECALC) and save_loc.exists():
        data = np.load(str(save_loc))
        spec_max_r, res , Width_vec_i  = data['spec_max_r'], data['res'], data['Width_vec_i']
        data = None

    else:
        res=spectrum(H_m,wlist,c_ops, (a*np.cosh(r)+a.dag()*np.sinh(r)),(a.dag()*np.cosh(r)+a*np.sinh(r)))
        spec_max_r=wlist[np.argmax(res)]
        w_0= spec_max_r    
        def func_dephase(w,sig):
            return (sig)**2/((w-w_0)**2+(sig)**2)
        popt, pcov = curve_fit(func_dephase, wlist, res/max(res), maxfev=5000) 
        Width_vec_i=abs((popt[0])/(2*np.pi)) 
        np.savez(str(save_loc), spec_max_r = spec_max_r, res = res ,Width_vec_i=Width_vec_i)

    return spec_max_r, res , Width_vec_i


def Calc_spec_spin(save_loc,H_m,wlist,c_ops,sx,r,RECALC):
    spec_max_r = None
    res = None
    Width_vec_i = None

    if (not RECALC) and save_loc.exists():
        data = np.load(str(save_loc))
        spec_max_r, res , Width_vec_i  = data['spec_max_r'], data['res'], data['Width_vec_i']
        data = None

    else:
        res=spectrum(H_m,wlist,c_ops, sx,sx)
        spec_max_r=wlist[np.argmax(res/(max(res)))]
        w_0= spec_max_r    
        def func_dephase(w,sig):
            return (sig)**2/((w-w_0)**2+(sig)**2)
        popt, pcov = curve_fit(func_dephase, wlist, res/max(res), maxfev=5000) 
        Width_vec_i=abs((popt[0])/(2*np.pi)) 
        np.savez(str(save_loc), spec_max_r = spec_max_r, res = res ,Width_vec_i=Width_vec_i)

    return spec_max_r, res , Width_vec_i


def Derivative_func(fx):
    dx=abs(fx[0]-fx[1])
    fx_tag = (fx[1:]-fx[0:-1])/dx
    return fx_tag


def find_zero_arg(fx):
    vec_arg0 = []
    i=0
    len_fx  =len(fx)
    for j in range(len_fx-1):
        if(fx[j]>=0 and fx[j+1]<0):
            vec_arg0.append(j)
            continue
        if(fx[j]<=0 and fx[j+1]>0):
            vec_arg0.append(j)
            continue
        else:
            continue   
    return np.array(vec_arg0)

def E_of_Nexpect(N,kappa,g, omega_r,r,gamma,n):

    Omega_c= omega_r/np.cosh(2*r) 
    a =tensor(destroy(N))
    H_m = Omega_c * a.dag() * a 
    c_ops = [np.sqrt(kappa*(n+1)) * (a*np.cosh(r)+a.dag()*np.sinh(r)), np.sqrt(kappa*(n)) * (a.dag()*np.cosh(r)+a*np.sinh(r))]  # Collapse Operators
    
    rho = steadystate(H_m,c_ops)
    N_expect = ((a.dag()*np.cosh(r)+a*np.sinh(r))* (a*np.cosh(r)+a.dag()*np.sinh(r))*rho).tr()
    
    return  -2*(g*np.exp(r)*np.sqrt(N_expect)-g*np.exp(r)*np.sqrt(N_expect+1))

def Photon_Number(N,kappa,g, omega_r,r,gamma,n):

    Omega_c= omega_r/np.cosh(2*r) 
    a =tensor(destroy(N))
    H_m = Omega_c * a.dag() * a 
    c_ops = [np.sqrt(kappa*(n+1)) * (a*np.cosh(r)+a.dag()*np.sinh(r)), np.sqrt(kappa*(n)) * (a.dag()*np.cosh(r)+a*np.sinh(r))]  # Collapse Operators
    
    rho = steadystate(H_m,c_ops)
    N_expect = ((a.dag()*np.cosh(r)+a*np.sinh(r))* (a*np.cosh(r)+a.dag()*np.sinh(r))*rho).tr()
    
    return  N_expect

def Lorentzian_func(w_0,wlist,width):

    def func_dephase(w,sig):
            return (sig)**2/((w-w_0)**2+(sig)**2)

    return func_dephase(wlist,width)


def Squeezing_operators(omega_r_tilde,omega_q_tilde,r,a,sz,sm,g,kappa,n):

    Omega_c= omega_r_tilde/np.cosh(2*r) 
    H_m = Omega_c * a.dag() * a + (0.5*omega_q_tilde) * sz +g*0.5*np.exp(r)*(a+a.dag())*(sm+sm.dag())-g*0.5*np.exp(-r)*(a.dag()-a)*(sm.dag()-sm)  #coupling- ' on'
    c_ops = [np.sqrt(kappa*(n+1)) * (a*np.cosh(r)+a.dag()*np.sinh(r)), np.sqrt(kappa*(n)) * (a.dag()*np.cosh(r)+a*np.sinh(r))]  # Collapse Operators
    
    return Omega_c,H_m,c_ops


def find_r_zoom(plt_max,omega_q_tilde):

    max_up = plt_max[plt_max>=(omega_q_tilde)/(2*np.pi)]   ; arg_max_up = np.where(plt_max==max_up[-1])
    max_down = plt_max[plt_max<(omega_q_tilde)/(2*np.pi)] ; arg_max_down = np.where(plt_max==max_down[0])

    return max_up,max_down,arg_max_up[0][0],arg_max_down[0][0]

def find_min_max(f_tag,arg_0_vec):

    min_vec=[] ; max_vec= []
    for arg_0 in arg_0_vec:
        if(f_tag[arg_0-1]>0 and f_tag[arg_0+1]<0):
            max_vec.append(arg_0)
        if(f_tag[arg_0-1]<0 and f_tag[arg_0+1]>0):
            min_vec.append(arg_0)

    return min_vec,max_vec