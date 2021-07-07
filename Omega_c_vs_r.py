
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


plt.rcParams["mathtext.fontset"]="cm"

#variables

N=30

r=np.linspace(0,1.2,100)

plt.plot(r,1/np.cosh(2*r),Linewidth = 3,Linestyle = 'dashed',color='darkred')
plt.ylabel(r'${\rm Frequency}$ $\left[{\rm  MHz}\right] $',fontsize='24')
plt.xlabel(r'$r$',fontsize='24')
plt.tick_params(axis='x', labelsize=18 )
plt.tick_params(axis='y', labelsize=18 )
plt.ylim([0,1.2])
fig = plt.gcf()
fig.set_size_inches(6,4)
fig.set_tight_layout(True)
plt.show()  

