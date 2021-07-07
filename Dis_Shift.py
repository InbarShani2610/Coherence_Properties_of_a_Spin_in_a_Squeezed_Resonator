
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
plt.rcParams["font.family"]="serif"
plt.rcParams["mathtext.fontset"]="cm"


RECALC = False
# RECALC = True


#variables

N=30
sz= tensor(qeye(N),sigmaz())
sx=tensor(qeye(N),sigmax())
sm=tensor(qeye(N),(sigmax()-1j*sigmay())/2)
a =tensor(destroy(N),qeye(2))
omega_r_tilde_vec   = np.array([1])*10**(-3) *2*np.pi 
kappa_vec = np.array([40])*10**(-6)*2*np.pi #20,10##  80,10,20,40 ,10,20,80,160
g=5*10**(-6)*2*np.pi   ## 5, 2.5, 1,0.5

resolution=  300000  
det_vec_0 =np.linspace(0.5,8,25)*100
det_vec_add =np.linspace(8.1,9.5,5)*100

plt.rcParams["mathtext.fontset"]="cm"

gamma=0*2*np.pi
n=0

for kappa in kappa_vec:

    if(kappa==80*10**(-6)*2*np.pi or kappa==40*10**(-6)*2*np.pi):
        det_vec=det_vec_0
        det_vec=[400]#[det_vec[13]]
        i=0
        for omega_r_tilde in omega_r_tilde_vec:
            # coupling_new =[]
            # coupling_new_error = []
            j=0
            for det in det_vec:   
                colormap_plot = [] 
                omega_q_tilde =omega_r_tilde-det*10**(-6)*2*np.pi 
                wlist=np.linspace(-omega_q_tilde-15*g,-omega_q_tilde+15*g,resolution)
                r_check = 0.5*np.arccosh(omega_r_tilde/omega_q_tilde) 
                r_vec =np.concatenate([np.linspace(0,1,60) ,np.arange(1+1/60,1.2,1/60),np.arange(1.2+1/60,1.4,1/60)])
                


                plt_max=[] 
                Shift_SW  =[]        
                for r in r_vec:
                    print('Probe r is ' ,r)  
                    l=omega_r_tilde*np.tanh(2*r)        
                    Omega_c= omega_r_tilde/np.cosh(2*r) 
                    H_m = Omega_c * a.dag() * a + (0.5*omega_q_tilde) * sz +g*0.5*np.exp(r)*(a+a.dag())*(sm+sm.dag())-g*0.5*np.exp(-r)*(a.dag()-a)*(sm.dag()-sm)  #coupling- ' on'
                    c_ops = [np.sqrt(kappa*(n+1)) * (a*np.cosh(r)+a.dag()*np.sinh(r)), np.sqrt(kappa*(n)) * (a.dag()*np.cosh(r)+a*np.sinh(r))]  # Collapse Operators

                    save_loc= Save_func('All_spin',wlist[0],wlist[-1],resolution,N,omega_r_tilde,omega_q_tilde,kappa,g,r,gamma)
                    spec_max_r, res , Width_vec_i = Calc_spec_spin(save_loc,H_m,wlist,c_ops,sx,r,RECALC)
                    # plt.plot(wlist+omega_q_tilde,res/(max(res)))
                    # plt.show()
                    plt_max.append((abs(spec_max_r)-omega_q_tilde)/(2*np.pi))
                    Shift_SW.append(((g**2)/(omega_q_tilde**2-omega_r_tilde**2+l**2))*((4*l**2*omega_q_tilde)/(4*omega_r_tilde**2+kappa**2-4*l**2)+omega_r_tilde+omega_q_tilde))
                plt_max= np.array(plt_max) ; Shift_SW= (np.array(Shift_SW ))
                plt.scatter(r_vec,plt_max*10**6,linewidth=2,marker='o',label='Numeric',color = 'teal')
                plt.plot(r_vec,Shift_SW*10**6/(2*np.pi),linewidth=3,color='mediumvioletred',label='SW')
                # plt.axvline(x=0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2-2500*g**2))),linewidth=2.5,linestyle='--',color='orangered')
                # plt.axvline(x=0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2+2500*g**2))),linewidth=2.5,linestyle='--',color='orangered')

                plt.grid(b=True,linestyle='dotted',linewidth=1.5)
                plt.xlabel(r'$r$',fontsize='28')
                plt.ylabel(r'$\delta \tilde{\omega}_s$ $ {\rm [kHz]}$',fontsize='28')
                plt.ylim([-2,2])
                # plt.yticks([-2,-1,0,1,2])
                plt.tick_params(axis='x', labelsize=20 )
                plt.tick_params(axis='y', labelsize=20 )
                # plt.fill_betweenx([-2,2],0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2+2500*g**2))),\
                    # 0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2-2500*g**2))),color='orangered',hatch = '///',facecolor = 'grey',alpha=0.25)
                plt.fill_betweenx([-2,2],0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2+2500*g**2))),\
                    0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2-2500*g**2))),facecolor = 'grey',alpha=0.25)

                plt.legend(loc='upper left', prop={'size': 18})
                fig = plt.gcf()
                fig.set_size_inches(8,5)
                fig.set_tight_layout(True)
     
  
plt.show()