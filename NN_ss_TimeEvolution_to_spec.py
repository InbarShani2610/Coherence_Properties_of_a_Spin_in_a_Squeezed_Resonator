
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from matplotlib import interactive
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from pathlib import Path
import math



N_vec=[30,90,150]

delta_c = 0.5  *2*np.pi

r_vec =np.linspace(0,2,40)
# lambda_t_vec=  delta_c * np.tanh(2*r_vec)
kappa = 0.1 *2*np.pi
tau_list = np.linspace(0,15,600)
lambda_t_vec=delta_c*np.tanh(2*r_vec)
dt=abs(tau_list[0]-tau_list[1])


Integral_0_inf= lambda kappa,delta_c,l: (2*l**2*(4*delta_c**2+kappa**2)*(4*delta_c**2+5*kappa**2-4*l**2))/(kappa*(4*delta_c**2+kappa**2-4*l**2)**3)

for N in N_vec :
    sol=[]
    for r in r_vec:
        a = destroy(N)   
        lambda_t = delta_c * np.tanh(2*r)
        ## Mathematica analytic function
        adaada_sol_math = lambda  tau , k, d, l:  (l**2*(\
            2*d*(d-1j*k)*np.sqrt(d**2-l**2)*np.exp(-k*tau)*(4*d**2+k**2-4*l**2)-\
            (np.sqrt((d-l)*(d+l))-d)*(-4*d**4+4*d**3*(np.sqrt((d-l)*(d+l))-1j*k)+d**2*(8*l**2-3*k**2)-\
                d*(k**2*np.sqrt(d**2-l**2)+4*l**2*np.sqrt(d**2-l**2)-4j*k*l**2)+1j*k**3*np.sqrt((d-l)*(d+l))+3*k**2*l**2-4*l**4)*np.exp(tau*(2j*np.sqrt(d**2-l**2)-k))+\
                    (np.sqrt((d-l)*(d+l))+d)*(4*d**4+4*d**3*(np.sqrt((d-l)*(d+l))+1j*k)+d**2*(3*k**2-8*l**2)\
                    +1j*k**3*np.sqrt((d-l)*(d+l))-d*(k**2*np.sqrt((d-l)*(d+l))+4*l**2*np.sqrt((d-l)*(d+l))+4j*k*l**2)-3*k**2*l**2 +4*l**4)*np.exp(-tau*(2j*np.sqrt(d**2-l**2)+k))\
                        +8*l**2*((d-l)*(d+l))**1.5))/(2*((d-l)*(d+l))**1.5*(4*d**2+k**2-4*l**2)**2) -((4*l**4)/(4*d**2+k**2-4*l**2)**2)

        NN_sol_ss_tau=adaada_sol_math(tau_list,kappa,delta_c,lambda_t)
        ## Time  Evolution

        H = delta_c * a.dag() *a - 0.5 * lambda_t * ( a * a + a.dag() * a.dag() )
        c_ops = [np.sqrt(kappa) * a ]
        rho_steady_state = steadystate(H,c_ops)
        initial_state =  (a.dag()*a*rho_steady_state)
        res = mesolve(H,initial_state,tau_list,c_ops,[])

        final_state =[]
        for i in range(len(tau_list)):
            final_state.append((a.dag()*a*res.states[i]).tr()-(a.dag()*a*res.states[-1]).tr())  # 
        print(r)
        sol=np.append(sol,sum(2*dt*np.real(final_state)))
        # plt.plot(tau_list,np.real(final_state),marker='v',label='Re Numerical value')
        # # plt.plot(tau_list,np.imag(final_state),marker='v',label='Im Numerical value')
        # plt.plot(tau_list,np.real(NN_sol_ss_tau),marker='o', label='Re Analytical Expression')
        # # plt.plot(tau_list,np.imag(NN_sol_ss_tau),marker='o', label=' Im Analytical Expression')
        # plt.title(r'$a^\dagger a a^\dagger a$ - Time difference $\tau$ Using Time Evolution')
        # plt.grid(b=True,linestyle='dotted',linewidth=1)
        # plt.xlabel(r'$\tau$')
        # plt.ylabel(r'$\left< a^\dagger a  a^\dagger a \right>$')
        # props = dict(boxstyle='round', facecolor='tab:orange', alpha=0.5, linewidth=2)
        # plt.legend()
        # plt.text(10,60,
        #     '''N={0:d}
        #     $\\delta_c={1:.2f}$
        #     $\\kappa={2:.3f}$
        #     $\\lambda={3:.2f}$'''.format(N,delta_c,kappa,lambda_t),fontsize= 12,bbox=props)
        # plt.show()
        
    plt.plot(r_vec,sol,marker='.',label='N='+str(N),linewidth=1)
    # plt.plot(r_vec,analytic_sol,label='analtic integral N='+str(N))

# plt.plot(r_vec,sum(2*np.real(adaada_sol_math(tau_list,kappa,delta_c,lambda_t_vec)),),label ='analytic integral')    
plt.plot(r_vec,Integral_0_inf(kappa,delta_c,lambda_t_vec) ,marker = '*', label ='Analytic',linewidth=1)
plt.legend()
plt.grid(b=True,linestyle='dotted',linewidth=1)
fig=plt.gcf()
fig.set_size_inches(6,3.42)
plt.show()