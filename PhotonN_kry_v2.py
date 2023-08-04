import numpy as np
import mpmath
from scipy import optimize
import scipy.integrate as integrate
import threading
import time
from time import sleep
import sys
import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'

mpmath.mp.dps = 15

mu0 = 4e-7*np.pi
c0 = 299792458
eps0 = 1/(c0**2*mu0)
hbar = 1.05457182e-34
kb = 1.380649e-23
mu = 1.60217663e-19*1.3e-9
Lc = 1.2e-4
Lr = 0.012
Np = 12
lp = Lc/Np
A = 1e-16

gamma = 1e12
gamma_normal = 1
gamma_c = 1e9/gamma
gamma_r = 1e10/gamma
Lambda_0 = 1e10/gamma
omega_0 = 1.6e14/gamma
N = 601
Temp_p = 400
Temp = 400
A = 1e-16
D = 2*mu**2/(hbar*eps0*A*Lc*gamma)
kb_normal = kb/(gamma**2)
hbar_normal = hbar/gamma


def fa(omega_n,T):
    return 1/(1+np.exp(-hbar_normal*omega_n/(kb_normal*T)))
def fb(omega_n,T):
    return np.exp(-hbar_normal*omega_n/(kb_normal*T))/(1+np.exp(-hbar_normal*omega_n/(kb_normal*T)))      

def Ndt(Nmati, modes, LOmega, T, Tp, nw, nn=1):
    
    range_a = range(nn); range_ok = range(nn, nn+nw)
    Kernel = np.zeros((nn,nw))
    Ndti = np.zeros(nn+nw)
    for n in range_a:
        for k in range_ok:
            Kernel[n,k-nn] = ((2*Nmati[n] - N)*Nmati[k] + Nmati[n])*LOmega[n,k-nn]
    Kernel_sn = np.sum(Kernel, axis=0)
    Kernel_sk = np.sum(Kernel, axis=1)
    for n in range_a:
        Lambda_n = Lambda_0*np.exp(hbar_normal*(omega_0-modes[n])/(kb_normal*Tp))
        Ndti[n] =  -D*Kernel_sk[n] + Lambda_n*(N-Nmati[n]) - gamma_r*(Nmati[n] - N*fb(modes[n],T))
    for k in range_ok:
        Ndti[k] = D*Kernel_sn[k-nn] - gamma_c*Nmati[k]
    return Ndti    


def Photon_n(Tp, omega_l, omega_u, n):
    if Tp == 1:
        emodes = np.load("Emodes_0_6pi_Tp1_v3.npz")
        print('load Tp = 1 data')
    elif Tp == 0.1:
        emodes = np.load("Emodes_0_6pi_Tp01_v3.npz")
        print('load Tp = 0.1 data')
    elif Tp == 0.4:
        emodes = np.load("Emodes_0_6pi_Tp04_v3.npz")
        print('load Tp = 0.4 data')
    elif Tp == 0.01:
        emodes = np.load("Emodes_0_6pi_Tp001_v3.npz")
        print('load Tp = 0.01 data')
    else:
        emodes = np.load("Emodes_0_6pi_miscs.npz")
        print('load Tp = miscs data')

    ks = emodes["ks"]
    G_in = emodes["G_in"]
    G_out = emodes["G_out"]
    Omega_k = ks/lp*c0/gamma
#    Nki = np.zeros((len(Omega_k)))
#    mean_nk = []
#    for omega in Omega_k:
#        mean_nk.append(1/(np.exp(hbar*omega/(kb*Temp))-1))
#    Nki = np.array(mean_nk)*3
    if Tp == 1:
        nit = np.load("Photon_n_0_6pi_Tp1_n50_kry_sp.npz")
        Nki = nit["Nkt"]
    elif Tp == 0.4:
        nit = np.load("Photon_n_0_6pi_Tp04_n500_A1e-16_mt.npz")
        Nki = nit["Nkt"][-1]
    elif Tp == 0.1:
        nit = np.load("Photon_n_0_6pi_Tp04_n500_kry.npz")
        Nki = nit["Nkt"]
    elif Tp == 0.01:
        nit = np.load("Photon_n_0_6pi_Tp01_n500_kry.npz")
        Nki = nit["Nkt"]
                      
    omega_n = np.linspace(omega_l, omega_u, n)/gamma
    Nbni = np.zeros(len(omega_n))
    nw = len(Omega_k)

    Ni = np.hstack((Nbni, Nki))
    modes = np.hstack((omega_n, Omega_k, G_in)) 
    
    LOmega=[]
    for omega in omega_n:
        LOmega.append(1/(1+((omega*np.ones(len(Omega_k)) - Omega_k)/gamma_normal)**2)*Omega_k*G_in)
    LOmega = np.array(LOmega)
    
    start_time = time.time()
    sol = optimize.newton_krylov(lambda ni: Ndt(ni, modes, LOmega, Temp, Temp_p, nw=nw, nn=n), Ni)
    end_time = time.time()
    print(f"The time used for Tp={Tp} is: {end_time-start_time}")

    Nb = sol[:n]
    Nk = sol[n:]
    Na = 1-Nb

    if Tp == 1:
        np.savez_compressed("Photon_n_0_6pi_Tp1_n500_kry_sp", Nat = (N-Nb), Nbt = Nb, Nkt = Nk)
        print('save Tp = 1 photon number data')
    elif Tp == 0.1:
        np.savez_compressed("Photon_n_0_6pi_Tp01_n500_kry_sp", Nat = (N-Nb), Nbt = Nb, Nkt = Nk)
        print('save Tp = 0.1 photon number data')
    elif Tp == 0.4:
        np.savez_compressed("Photon_n_0_6pi_Tp04_n500_kry_sp", Nat = (N-Nb), Nbt = Nb, Nkt = Nk)
        print('save Tp = 0.4 photon number data')
    elif Tp == 0.01:
        np.savez_compressed("Photon_n_0_6pi_Tp001_n500_kry_sp", Nat = (N-Nb), Nbt = Nb, Nkt = Nk)
        print('save Tp = 0.01 photon number data')
    else:
        np.savez_compressed("Photon_n_0_6pi_miscs", Nat = (N-Nb), Nbt = Nb, Nkt = Nk)
        print('save Tp = misc photon number data')



Tpl = [1]

for Tp in Tpl:
    Photon_n(Tp, 0.1, 5.2e14, 500)
    

#if __name__ == "__main__":
    
#    t1 = threading.Thread(target=Photon_n, args=(0.01, 0.1, 5.2e14, 500, 1.5e-11))
#    t2 = threading.Thread(target=Photon_n, args=(0.1, 0.1, 5.2e14, 500, 1.5e-11)) 

