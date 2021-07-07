
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

RECALC = False
# RECALC = True


#variables

N=40

omega_r_tilde_vec   = np.array([1])*10**(-3) *2*np.pi 
kappa_vec = np.array([20,40,80,160,240])*10**(-6)*2*np.pi #20,10##  80,10,20,40 ,10,20,80,160
g=5*10**(-6)*2*np.pi   ## 5, 2.5, 1,0.5

resolution=  300000  
det_vec_0 =np.linspace(0.5,8,25)*100
det_vec_add1 =np.linspace(8,8.1,3)*100
det_vec_add2 =np.linspace(8.1,9.5,5)*100
det_vec=np.concatenate([det_vec_0,det_vec_add1,det_vec_add2[0:-2]])

plt.rcParams["mathtext.fontset"]="cm"

gamma=0*2*np.pi
n=0

i=0
markers_vec = ['*','^','o','s','D']
for kappa in kappa_vec:    
    for omega_r_tilde in omega_r_tilde_vec:
        PhotonNumber_vec =[]
        Analytic =[]
        r_vec = np.linspace(0.8,1.2,25)            
        for r in r_vec:
            l=omega_r_tilde*np.tanh(2*r)
            print('Probe r is ' ,r)          
            PhotonNumber_vec.append(Photon_Number(N,kappa,g, omega_r_tilde,r,gamma,0))
            Analytic.append(2*l**2/(4*omega_r_tilde**2 + kappa**2-4*l**2 ))
        plt.plot(r_vec,PhotonNumber_vec,linewidth=1,marker=markers_vec[i],label= r'$\kappa=$ {:.0f}'.format(kappa*10**6/(2*np.pi)))
    i=i+1


plt.ylabel(r'$\left<a^\dagger a\right>$',fontsize='24')
plt.xlabel(r'$r$',fontsize='24')
plt.tick_params(axis='x', labelsize=18 )
plt.tick_params(axis='y', labelsize=18 )
# plt.grid(b=True,linestyle='dotted',linewidth=1)
# plt.legend()
fig = plt.gcf()
fig.set_size_inches(6,4)
fig.set_tight_layout(True)
plt.show()  

