import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from matplotlib import interactive
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from Funcs import *


global RECALC



def jc_hamiltonian(r):
    N = 30                  # number of cavity fock states
    wc = 1
    wa = 1 # cavity and atom frequency
    delta_c=wc*np.cosh(2*r)
    gamma =  5*10**(-4) *delta_c   # cavity dissipation rate
    kappa =gamma
    g  =0.2 *gamma    # coupling strength
    
    # Jaynes-Cummings Hamiltonian
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), sigmam())
    sz = tensor(qeye(N),sigmaz())
    H_rabi =  wc * a.dag() * a + (wa*0.5) * sz + (g*0.5)*np.exp(r)* (a.dag() + a )*(sm+sm.dag())
    H_err=  (-g*0.5)*np.exp(-r)* (a.dag() - a )*(sm.dag()-sm)
    H =H_rabi+H_err

    # collapse operators
    #n_th = 0.01
    # c_ops = [np.sqrt(kappa * (1 + n_th)) * a, np.sqrt(kappa * n_th) * a.dag(), np.sqrt(gamma) * sm]
    # c_ops = [np.sqrt(kappa)* (a*np.cosh(r) +a.dag()*np.sinh(r)),np.sqrt(gamma)*sm]
    c_ops = [np.sqrt(kappa)*a,np.sqrt(gamma)*sm]
    return H,c_ops,a,sm,delta_c , kappa,gamma,g,wc,N


RECALC = False
plt.rcParams["font.family"]="serif"
plt.rcParams["mathtext.fontset"]="cm"



val_r =[0,0.5*np.log(100)]
ax=plt.axes()
color_vec=['blue','red']
marker_vec=['o','s']
i=0

for r in val_r:
    N = 30                  # number of cavity fock states
    wc = 1
    wa = 1 # cavity and atom frequency
    delta_c=wc*np.cosh(2*r)
    gamma =  5*10**(-4) *delta_c   # cavity dissipation rate
    kappa =gamma
    g  =0.2 *gamma    # coupling strength
    H,c_ops,a,sm,delta_c , kappa,gamma,g,wc,N = jc_hamiltonian(r)
    wlist = np.linspace(-wc-4*gamma, -wc+4*gamma, 100)
    # res = spectrum(H, wlist2, c_ops, a, a.dag())
    res = None
    save_loc= Save_func('Clerk_FigData1',wlist[0],wlist[-1],len(wlist),N,wc,wa,kappa,g,r,gamma)

    if (not RECALC) and save_loc.exists():
        data = np.load(str(save_loc))
        res =  data['res']
        data = None

    else:
        res = spectrum(H, wlist, c_ops, sm+sm.dag(), sm+sm.dag())
        np.savez(str(save_loc), res = res )

    ax.plot((wlist+wc)/(kappa) ,res/max(res),marker=marker_vec[i],markersize=10,color=color_vec[i],label= r'$ e^{2r}= $' + str(10*np.log10(np.exp(2*r)))+'dB',linewidth=4 )
    i=i+1

ax.set_xlabel(r'$(\omega-\delta_q)/\kappa$',fontsize=45, fontweight='bold')
# ax.set_tight_layout(True)
ax.set_ylabel(r'$S_s\left[\omega\right]$',fontsize=45,fontweight='bold')
ax.tick_params( axis='y', labelsize=35)
ax.set_xticks([-4,-2,0,2,4])
ax.tick_params( axis='x', labelsize=35)
ax.legend(loc='upper right', fontsize = 30)
ax.grid(b=True,linestyle='dotted',linewidth=2)
fig = plt.gcf()
fig.set_size_inches(15,8)
fig.set_tight_layout(True)
plt.show()
exit


