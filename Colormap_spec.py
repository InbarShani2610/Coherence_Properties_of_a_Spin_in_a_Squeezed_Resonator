
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

N=30
sz= tensor(qeye(N),sigmaz())
sx=tensor(qeye(N),sigmax())
sm=tensor(qeye(N),(sigmax()-1j*sigmay())/2)
a =tensor(destroy(N),qeye(2))
omega_r_tilde_vec   = np.array([1])*10**(-3) *2*np.pi 
kappa_vec = np.array([80])*10**(-6)*2*np.pi #20,10##  80,10,20,40 ,10,20,80,160
g=5*10**(-6)*2*np.pi   ## 5, 2.5, 1,0.5

resolution=  300000  
det_vec_0 =np.linspace(0.5,8,25)*100
det_vec_add =np.linspace(8.1,9.5,5)*100

plt.rcParams["mathtext.fontset"]="cm"

gamma=0*2*np.pi
n=0
# kappa_vec=[]
for kappa in kappa_vec:

    if(kappa==80*10**(-6)*2*np.pi or kappa==40*10**(-6)*2*np.pi):
        det_vec=det_vec_0
        det_vec=det_vec[[0,13,20]]
        # fig = plt.figure(constrained_layout=True )
        # spec = plt.GridSpec(ncols=len(det_vec), nrows=len(kappa_vec), figure=fig)
        # fig, ax = plt.subplots(ncols=len(det_vec),nrows=len(kappa_vec), figsize=(12, 6))
        fig, ax = plt.subplots(1,3, figsize=(18, 5))
        i=0
        pos= []
        for omega_r_tilde in omega_r_tilde_vec:
            # coupling_new =[]
            # coupling_new_error = []
            j=0
            for det in det_vec:   
                colormap_plot = [] 
                omega_q_tilde =omega_r_tilde-det*10**(-6)*2*np.pi 
                wlist=np.linspace(-omega_q_tilde-15*g,-omega_q_tilde+15*g,resolution)
                r_check = 0.5*np.arccosh(omega_r_tilde/omega_q_tilde) 
                r_vec =np.linspace(r_check-0.075,r_check+0.075,60)
                r_vec= r_vec[20:33]
                plt_r=[]; plt_max=[]           
                for r in r_vec:
                    print('Probe r is ' ,r)          
                    Omega_c= omega_r_tilde/np.cosh(2*r) 
                    H_m = Omega_c * a.dag() * a + (0.5*omega_q_tilde) * sz +g*0.5*np.exp(r)*(a+a.dag())*(sm+sm.dag())-g*0.5*np.exp(-r)*(a.dag()-a)*(sm.dag()-sm)  #coupling- ' on'
                    c_ops = [np.sqrt(kappa*(n+1)) * (a*np.cosh(r)+a.dag()*np.sinh(r)), np.sqrt(kappa*(n)) * (a.dag()*np.cosh(r)+a*np.sinh(r))]  # Collapse Operators

                    save_loc= Save_func('All',wlist[0],wlist[-1],resolution,N,omega_r_tilde,omega_q_tilde,kappa,g,r,gamma)
                    spec_max_r, res , Width_vec_i = Calc_spec(save_loc,H_m,wlist,c_ops,a,r,RECALC)

                    plt_max.append(abs((abs(spec_max_r)))/(2*np.pi))
                    
                plt_max= np.array(plt_max)
                max_up = plt_max[plt_max>=(omega_q_tilde+g)/(2*np.pi)]   ; arg_max_up = np.where(plt_max==max_up[-1])
                max_down = plt_max[plt_max<=(omega_q_tilde-g)/(2*np.pi)] ; arg_max_down = np.where(plt_max==max_down[0])
               
                r_zoom_vec=np.linspace(r_vec[arg_max_up],r_vec[arg_max_down],25)
                plt_max_new = []


                r_vec_colorplot = list(np.concatenate([r_vec,[]]))
                r_vec_colorplot.sort()
                r_vec_cmap_nodups=[]
                [r_vec_cmap_nodups.append(x) for x in r_vec_colorplot if x not in r_vec_cmap_nodups]
                print(r_vec_cmap_nodups)


                r_vec_cmap_nodups = r_vec_cmap_nodups[-6:]
                r_center = abs((r_vec_cmap_nodups[0]+r_vec_cmap_nodups[-1]))/2
                print(r_center)
                r_vec_cmap_nodups = np.linspace(r_center-0.015,r_center+0.015,20)
                r_vec_add_2 = np.linspace(r_center+0.015,r_center+0.03,10)
                r_vec_add_1 = np.linspace(r_center-0.03,r_center-0.015,10)
                r_vec_cmap_nodups = np.concatenate([r_vec_add_1,r_vec_cmap_nodups,r_vec_add_2])
                for r in r_vec_cmap_nodups:
                    print('Probe r is ' ,r)          
                    Omega_c= omega_r_tilde/np.cosh(2*r) 
                    H_m = Omega_c * a.dag() * a + (0.5*omega_q_tilde) * sz +g*0.5*np.exp(r)*(a+a.dag())*(sm+sm.dag())-g*0.5*np.exp(-r)*(a.dag()-a)*(sm.dag()-sm)  #coupling- ' on'
                    c_ops = [np.sqrt(kappa*(n+1)) * (a*np.cosh(r)+a.dag()*np.sinh(r)), np.sqrt(kappa*(n)) * (a.dag()*np.cosh(r)+a*np.sinh(r))]  # Collapse Operators

                    save_loc= Save_func('All',wlist[0],wlist[-1],resolution,N,omega_r_tilde,omega_q_tilde,kappa,g,r,gamma)
                    spec_max_r, res , Width_vec_i = Calc_spec(save_loc,H_m,wlist,c_ops,a,r,RECALC)
                    res=res/max(res)
                    minw=4 *10**4  
                    maxw=26*10**4 
                    
                    colormap_plot.append(res[minw:maxw])

                wlist_detuned=(abs(omega_q_tilde)-abs(wlist[minw:maxw]))/(2*np.pi)*10**6 ### changing to kHz
                colormap_plot=np.array(colormap_plot).transpose()
                # colormap_plot=colormap_plot[:,-5:]
                # axij=fig.add_subplot()
                
                pos.append(ax[i].imshow(colormap_plot,aspect='auto',interpolation='nearest',\
                     extent=[r_vec_cmap_nodups[0],r_vec_cmap_nodups[-1],wlist_detuned[0],wlist_detuned[-1]]\
                         ,origin = 'upper',cmap=plt.cm.viridis))
                ax[i].plot(r_vec_cmap_nodups,(omega_r_tilde)*10**6/((2*np.pi)*np.cosh(2*np.array(r_vec_cmap_nodups)))-omega_q_tilde*10**6/(2*np.pi)\
                    ,color='red',linestyle='dashed',linewidth=2 )
                # ax[i].set_title(r'$r_c = $ {:.3f} '.format(r_check),fontsize=18)
                ax[i].tick_params(axis='x', labelsize=20 )
                ax[i].tick_params(axis='y', labelsize=20 )
                # ax[i].set_xticks([])
                # ax[i].set_xlabel(r'$r$',fontsize=24,fontweight='bold')
                # ax[i].set_ylabel(''  if i != 0 else r'$\Delta$ [kHz]',fontsize=24,fontweight='bold')  
                # if i!=0 :
                    # ax[i].set_yticks([])
                ax[i].set_ylabel(''  if i != 0 else r'$\omega-\tilde{\omega}_s$ ${\rm [kHz]}$',fontsize=30,fontweight='bold')   
                ax[i].set_xlabel( r'$r$',fontsize=30,fontweight='bold')      
               
                # axij.set_ylim([-50,50])
                i=i+1
    fig.colorbar(pos[0],ax=ax[0:3],aspect=15,label='Power Spectrum',orientation='vertical')
    # fig.set_tight_layout(True)
    p0 = ax[0].get_position().get_points().flatten()
    p1 = ax[1].get_position().get_points().flatten()
    p2 = ax[2].get_position().get_points().flatten()

    # ax_cbar = fig.add_axes((0,.0,1,.05))
    # cbar = fig.colorbar(pos[0],ax=ax[0:3],aspect=30, orientation='horizontal',ticks=np.round(np.linspace(0.07,1,5),1))        
    # cbar.ax.tick_params(labelsize=18) 
    # cbar.set_label(label='Power Spectrum',fontsize=20,fontweight='bold')
           
    plt.show()