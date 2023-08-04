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
gamma_c = 1e9
gamma_r = 1e13
Lambda_0 = 1e10
omega_0 = 1.6e14
N = 601
Temp_p = 400
Temp = 400
D = 2*mu**2/(hbar*eps0*A*Lc*gamma)

def fa(omega_n,T):
    return 1/(1+np.exp(-hbar*omega_n/(kb*T)))
def fb(omega_n,T):
    return np.exp(-hbar*omega_n/(kb*T))/(1+np.exp(-hbar*omega_n/(kb*T)))   

def Ndtv2(Nmati, modes, LOmega, ksum_r, T, Tp, nw, nn=1):
    range_a = range(nn); range_ok = range(nn, nn+nw)
    Kernel = np.zeros((nn,nw))
    Ndti = np.zeros(2*nn+nw)
    for n in range_a:
        for k in ksum_r[n]:
            Kernel[n,k] = ((Nmati[nn+n] - Nmati[n])*Nmati[2*nn+k] + Nmati[nn+n])*LOmega[n,k]
    Kernel_sn = np.sum(Kernel, axis=0)
    Kernel_sk = np.sum(Kernel, axis=1)
    for n in range_a:
        Lambda_n = Lambda_0*np.exp(hbar*(omega_0-modes[n])/(kb*Tp))
        Ndti[n] = D*Kernel_sk[n] - Lambda_n*Nmati[n] - gamma_r*(Nmati[n] - N*fa(modes[n],T))
        Ndti[nn+n] =  -D*Kernel_sk[n] + Lambda_n*Nmati[n] - gamma_r*(Nmati[nn+n] - N*fb(modes[n],T))
    for k in range_ok:
        Ndti[nn+k] = D*Kernel_sn[k-nn] - gamma_c*Nmati[nn+k]
    return Ndti    
 

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
        Lambda_n = Lambda_0*np.exp(hbar*(omega_0-modes[n])/(kb*Tp))
        Ndti[n] =  -D*Kernel_sk[n] + Lambda_n*(N-Nmati[n]) - gamma_r*(Nmati[n] - N*fb(modes[n],T))
    for k in range_ok:
        Ndti[k] = D*Kernel_sn[k-nn] - gamma_c*Nmati[k]
    return Ndti   


def Photon_n(Tp, omega_l, omega_u, n, tbound, rcut=1e-4):
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
    Omega_k = ks/lp*c0
    Nki = np.zeros((len(Omega_k)))
#    mean_nk = []
#    for omega in Omega_k:
#        mean_nk.append(1/(np.exp(hbar*omega/(kb*Temp))-1))
#    Nki = mean_nk
    omega_n = np.linspace(omega_l, omega_u, n)
    Nani = np.ones(len(omega_n))*N
    Nbni = np.zeros(len(omega_n))
    nw = len(Omega_k)

    Ni = np.hstack((Nbni, Nki))
    modes = np.hstack((omega_n, Omega_k, G_in)) 
    
    LOmega=[]
    for omega in omega_n:
        LOmega.append(1/(1+((omega*np.ones(len(Omega_k)) - Omega_k)/gamma)**2)*Omega_k*G_in)
    LOmega = np.array(LOmega)
#    ksum_range=[]
#    for LOmega_k in LOmega:
#        ksum = [i for i,v in enumerate(LOmega_k/np.max(LOmega_k)) if v>rcut]
#        ksum_range.append(ksum)

    nsol = integrate.RK45(lambda t, ni: Ndt(ni, modes, LOmega, Temp, Temp_p, nw=nw, nn=n), 
                          t0=0, y0=Ni, t_bound = tbound)

    start_time = time.time()
    t_values = []
    y_values = []
    i = 0
    while nsol.status != 'finished':
        # get solution step state
        nsol.step()
        i += 1
        if (i%20) == 0:
            t_values.append(nsol.t)
            y_values.append(nsol.y)
            print(f"The time is: {nsol.t}")
    t_values.append(nsol.t)
    y_values.append(nsol.y)
    y_values = np.array(y_values)
    Nb = y_values[...,:n]
    Nk = y_values[...,n:]
    end_time = time.time()
    print(f'\n The time used: {end_time-start_time}')

    if Tp == 1:
        np.savez_compressed("Photon_n_0_6pi_Tp1_n500_t1e-9", Nat = (N-Nb), Nbt = Nb, Nkt = Nk, t = t_values)
        print('save Tp = 1 photon number data')
    elif Tp == 0.1:
        np.savez_compressed("Photon_n_0_6pi_Tp01_n500_t1e-9", Nat = (N-Nb), Nbt = Nb, Nkt = Nk, t = t_values)
        print('save Tp = 0.1 photon number data')
    elif Tp == 0.4:
        np.savez_compressed("Photon_n_0_6pi_Tp04_n500_t1e-9", Nat = (N-Nb), Nbt = Nb, Nkt = Nk, t = t_values)
        print('save Tp = 0.4 photon number data')
    elif Tp == 0.01:
        np.savez_compressed("Photon_n_0_6pi_Tp001_n500_t1e-9", Nat = (N-Nb), Nbt = Nb, Nkt = Nk, t = t_values)
        print('save Tp = 0.01 photon number data')
    else:
        np.savez_compressed("Photon_n_0_6pi_miscs_sp", Nat = (N-Nb), Nbt = Nb, Nkt = Nk, t = t_values)
        print('save Tp = misc photon number data')



Tpl = [0.4, 0.01]

for Tp in Tpl:
    Photon_n(Tp, 0.1, 5.2e14, 500, 1e-11)
    

#if __name__ == "__main__":
    
#    t1 = threading.Thread(target=Photon_n, args=(0.01, 0.1, 5.2e14, 500, 1.5e-11))
#    t2 = threading.Thread(target=Photon_n, args=(0.1, 0.1, 5.2e14, 500, 1.5e-11)) 

