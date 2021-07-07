
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
        print(det_vec)
        i=0
        for omega_r_tilde in omega_r_tilde_vec:
            # coupling_new =[]
            # coupling_new_error = []
            j=0
            for det in det_vec:   
              
                omega_q_tilde =omega_r_tilde-det*10**(-6)*2*np.pi 
                wlist=np.linspace(-omega_q_tilde-15*g,-omega_q_tilde+15*g,resolution)
                r_check = 0.5*np.arccosh(omega_r_tilde/omega_q_tilde) 
                r_vec =np.concatenate([np.linspace(0,1,60) ,np.arange(1+1/60,1.2,1/60), np.arange(1.2+1/60,1.4,1/60)])
                

                plt_max=[] 
                Shift_SW  =[] 
                GammaPurcell = [] 
                Width_numeric = []   
                GammaPhoton = []   
                GammaPhoton_Check = []
                for r in r_vec:
                    print('Probe r is ' ,r)  
                    l=omega_r_tilde*np.tanh(2*r) 
                    Omega_c= omega_r_tilde/np.cosh(2*r)        
                    Delta_square =omega_q_tilde**2-omega_r_tilde**2+l**2
                    C1=g*(omega_r_tilde+omega_q_tilde)/Delta_square
                    C2=g*l/Delta_square
                    GammaPurcell.append(kappa*(C1**2+C2**2))
                    delta_c=omega_r_tilde
                    delta_q=omega_q_tilde
                    Analytic_photon_noise = (4* l**2* ((2* delta_c**2+kappa**2-2 *l**2)* (4 *delta_c**2+kappa**2-4* l**2)**2+4* delta_c *delta_q *(4* delta_c**2+3* kappa**2-4* l**2)* (4 *delta_c**2+kappa**2-4* l**2)\
                        +2* (4* delta_c**2+kappa**2)* delta_q**2* (4 *delta_c**2+5* kappa**2-4* l**2)))/(kappa* (4* delta_c**2+kappa**2-4* l**2)**3)
                    GammaPhoton.append(((g**4)/(delta_q**2-delta_c**2+l**2)**2)*Analytic_photon_noise)
                    GammaPhoton_Check.append(2*((g**4)/(delta_q-Omega_c)**2)*(1/(2*(Omega_c+delta_q))**2)*Analytic_photon_noise)
                    


                    H_m = Omega_c * a.dag() * a + (0.5*omega_q_tilde) * sz +g*0.5*np.exp(r)*(a+a.dag())*(sm+sm.dag())-g*0.5*np.exp(-r)*(a.dag()-a)*(sm.dag()-sm)  #coupling- ' on'
                    c_ops = [np.sqrt(kappa*(n+1)) * (a*np.cosh(r)+a.dag()*np.sinh(r)), np.sqrt(kappa*(n)) * (a.dag()*np.cosh(r)+a*np.sinh(r))]  # Collapse Operators

                    save_loc= Save_func('All_spin',wlist[0],wlist[-1],resolution,N,omega_r_tilde,omega_q_tilde,kappa,g,r,gamma)
                    spec_max_r, res , Width_vec_i = Calc_spec_spin(save_loc,H_m,wlist,c_ops,sx,r,RECALC)
                   
                   
                    # w_0= spec_max_r
                    # plt_max.append(abs((abs(spec_max_r)))/(2*np.pi))
                    # def func_dephase(w,sig):
                    #         return (sig)**2/((w-w_0)**2+(sig)**2)
                    # plt.plot(wlist,res/(max(res)))
                    # plt.plot(wlist,func_dephase(wlist,Width_vec_i*2*np.pi))  ## fit plot 
                    # plt.show()


                    Width_numeric.append(Width_vec_i*10**6)  ##already devided by two pi

                Width_numeric=np.array(Width_numeric)
                GammaPurcell=np.array(GammaPurcell)*10**6/(2*np.pi)
                GammaPhoton=np.array(GammaPhoton)*10**6/(2*np.pi)
                GammaPhoton_Check=np.array(GammaPhoton_Check)*10**6/(2*np.pi)
                # plt.plot(r_vec,plt_max*10**6,linewidth=2,label='Numeric')
                plt.plot(r_vec,Width_numeric,linewidth=2,color='teal')
                # plt.scatter(r_vec,0.5*GammaPurcell,linewidth=2,marker='d',color='mediumvioletred',label=r'SW')
                # plt.scatter(r_vec,GammaPhoton_Check,linewidth=2,marker='^',color='darkgoldenrod',label=r'Photon')
                plt.scatter(r_vec,0.5*GammaPurcell+np.array(GammaPhoton_Check),linewidth=2,marker='o'\
                    ,color='peru',label=r'$\frac{1}{2}\Gamma_{\rm purcell}+\Gamma_{\phi}^{\rm photon}$')
                # plt.axvline(x=0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2-2500*g**2))),linewidth=2.5,linestyle='--',color='orangered')
                # plt.axvline(x=0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2+2500*g**2))),linewidth=2.5,linestyle='--',color='orangered')

                plt.grid(b=True,linestyle='dotted',linewidth=1.5)
                plt.xlabel(r'$r$',fontsize='28')
                plt.ylabel(r'$\delta \tilde{\omega}_q$ $ {\rm [kHz]}$',fontsize='28')
                plt.ylabel(r'$\Gamma$ $ {\rm [kHz]}$',fontsize='28') # _{\rm purcell}
                plt.ylim([0,2])
                # plt.yticks([-2,-1,0,1,2])
                plt.tick_params(axis='x', labelsize=20 )
                plt.tick_params(axis='y', labelsize=20 )
                # plt.fill_betweenx([-2,2],0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2+2500*g**2))),\
                    # 0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2-2500*g**2))),color='orangered',hatch = '///',facecolor = 'grey',alpha=0.25)
                plt.fill_betweenx([-2,2],0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2+2500*g**2))),\
                    0.5*np.arccosh(omega_r_tilde/(np.sqrt(omega_q_tilde**2-2500*g**2))),facecolor = 'grey',alpha=0.25)
                plt.legend(loc='upper right', prop={'size': 18})
                fig = plt.gcf()
                fig.set_size_inches(8,5)
                fig.set_tight_layout(True)
     
  
plt.show()