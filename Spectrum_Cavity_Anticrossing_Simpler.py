
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from matplotlib import interactive
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from pathlib import Path
import math
from scipy.optimize import curve_fit
from Funcs import *

global RECALC
plt.rcParams["mathtext.fontset"]="cm"


RECALC = False
# RECALC = True


#  Operators
N=30  # Truncation parameter
sz= tensor(qeye(N),sigmaz())
sx=tensor(qeye(N),sigmax())
sm=tensor(qeye(N),(sigmax()-1j*sigmay())/2)
a =tensor(destroy(N),qeye(2))

# Variables
markers_vec = ['*','^','o','s','D']
omega_r_tilde  = 1*10**(-3) *2*np.pi 
kappa_vec = np.array([20,40,80,160,240])*10**(-6)*2*np.pi ## 20,40,80,160,240,500,1000      #np.array([20,40,80,160])*10**(-6)*2*np.pi #20,10##  80,10,20,40 ,10,20,80,160,240
g=5*10**(-6)*2*np.pi 
det_vec_0 =np.linspace(0.5,8,25)*100
det_vec_add1 =np.linspace(8,8.1,3)*100
det_vec_add2 =np.linspace(8.1,9.5,5)*100
det_vec=np.concatenate([det_vec_0,det_vec_add1,det_vec_add2[0:-2]])

gamma=0*2*np.pi
n=0 ## non-zero for thermal case

fig, ax = plt.subplots(2,1, figsize=(8, 12))
# Resolution for spectrum
resolution=  300000  
i=0
for kappa in kappa_vec:
    coupling_vec =[]
    contrast_vec =[]
    for det in det_vec:    
        omega_q_tilde =omega_r_tilde-det*10**(-6)*2*np.pi 
        wlist=np.linspace(-omega_q_tilde-kappa,-omega_q_tilde+kappa,resolution)
        r = 0.5*np.arccosh(omega_r_tilde/omega_q_tilde)    
        print(r)  
        Omega_c,H_m,c_ops =Squeezing_operators(omega_r_tilde,omega_q_tilde,r,a,sz,sm,g,kappa,n)
        save_loc= Save_func('All',wlist[0],wlist[-1],resolution,N,omega_r_tilde,omega_q_tilde,kappa,g,r,gamma)
        spec_max_r, res , Width_vec_i = Calc_spec(save_loc,H_m,wlist,c_ops,a,r,RECALC)
        # plt.plot(wlist,res/max(res))
        # plt.plot(wlist,Lorentzian_func(spec_max_r,wlist,Width_vec_i*2*np.pi))
        # plt.xlim([-0.006,-0.0059])
        # plt.show()
        res_normalized = res/max(res)
        res_tag = Derivative_func(res_normalized) 
        # plt.plot(wlist[0:-1],res_tag)
        # plt.show()
        zero_arg_vec = find_zero_arg(res_tag)
        vec_minima , vec_maxima = find_min_max(res_tag,zero_arg_vec)
        # plt.vlines(wlist[vec_maxima[0]],0,1,'r')
        # plt.vlines(wlist[vec_maxima[1]],0,1,'r')
        # plt.vlines(wlist[vec_minima[0]],0,1,'k')
        # plt.show()
        # plt.plot(wlist[(vec_maxima[0]-10):(vec_maxima[1]+10)] ,res_normalized[(vec_maxima[0]-10):(vec_maxima[1]+10)])
        # plt.show()
        if len(vec_maxima)==1:
            coupling_vec.append(0) 
            contrast_vec.append(0)
        else:
            coupling_vec.append(abs(min(wlist[vec_maxima[1]]-wlist[vec_minima[0]],wlist[vec_maxima[0]]-wlist[vec_minima[0]])))   
            contrast_vec.append(abs(min(res_normalized[[vec_maxima[1]]] -res_normalized[vec_minima[0]]\
                ,res_normalized[[vec_maxima[0]]] -res_normalized[vec_minima[0]])))
        # print(abs(zero_arg[0]-zero_arg[-1]))
    ax[0].plot(0.5*np.arccosh(omega_r_tilde/(omega_r_tilde-det_vec*10**(-6)*2*np.pi )),np.array(coupling_vec)*10**6/(2*np.pi),\
             marker=markers_vec[i],label=r'$\kappa=$ '+str(np.round(kappa*10**6/(2*np.pi)))+ r' kHz')
    ax[1].plot(0.5*np.arccosh(omega_r_tilde/(omega_r_tilde-det_vec*10**(-6)*2*np.pi )),np.array(contrast_vec)*100,\
                marker=markers_vec[i],label=r'$\kappa=$ '+str(np.round(kappa*10**6/(2*np.pi)))+ r' kHz')
    i=i+1
ax[0].set_ylabel(r' $ \chi$ ${\rm [kHz]} $ ',fontsize='28')
# ax[0].xlabel(r'$r_c$',fontsize='28')
ax[0].tick_params(axis='x', labelsize=22 )
ax[0].tick_params(axis='y', labelsize=22 )

ax[1].set_ylabel(r'$ {\rm Contrast }$ $\left[\%\right]$',fontsize='28')
ax[1].set_xlabel(r'$r_c$',fontsize='28')
ax[1].tick_params(axis='x', labelsize=22 )
ax[1].tick_params(axis='y', labelsize=22 )
# plt.legend(loc='lower left', prop={'size': 22})
plt.legend(loc='upper right', prop={'size': 22})
fig.set_tight_layout(True)
plt.show()  
  
