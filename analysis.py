import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from fit import *

def lorentzian(x, x0, gamma):
    return 1 / np.pi * gamma / ((x - x0)**2 + gamma**2) # gamma is the HWHM

def three_peaks(f,f1,f2,f3,A,Arel2,Arel3, gamma1, gamma2, gamma3,offset):
    return A * (lorentzian(f, f1, gamma1) + Arel2 * lorentzian(f, f2, gamma2) + Arel3 * lorentzian(f, f3, gamma3) ) + offset

def four_peaks(f, f1, f2, f3, f4, A, Arel2, Arel3, Arel4, gamma1, gamma2, gamma3, gamma4, offset):
    return A * (lorentzian(f, f1, gamma1) + Arel2 * lorentzian(f, f2, gamma2) + Arel3 * lorentzian(f, f3, gamma3) + Arel4 * lorentzian(f, f4, gamma4)) + offset

def six_peaks(f, f1, f2, f3, f4, f5, f6, A, Arel2, Arel3, Arel4, Arel5, Arel6, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, offset):
    return A * (lorentzian(f, f1, gamma1) + Arel2 * lorentzian(f, f2, gamma2) + Arel3 * lorentzian(f, f3, gamma3) + Arel4 * lorentzian(f, f4, gamma4) + Arel5 * lorentzian(f, f5, gamma5) + Arel6 * lorentzian(f, f6, gamma6)) + offset

dataset_name='24_11_18/100_seconds_80_deg_cooling_300_mA_illumination_low_pressure'
data=np.loadtxt(dataset_name+'.asc', skiprows=32, delimiter=',')
f_raw=data[:,0]
a=np.where(f_raw>940)[0][0]
b=np.where(f_raw<1000)[0][-1]
f=f_raw[a:b]
s=data[a:b,1]

f1=960.1
f2=971.5
f3=992.6
f4=980.6
f5=983.5
f6=995.5

A=1600
Arel2=2
Arel3=3
Arel4=0.5
Arel5=0.5
Arel6=3

gamma1=5
gamma2=10
gamma3=7
gamma4=10
gamma5=10
gamma6=10

#* Six peaks

# p0=[f1,f2,f3,f4,f5,f6,A,Arel2,Arel3,Arel4,Arel5,Arel6,gamma1,gamma2,gamma3,gamma4,gamma5,gamma6,250]
# names=['Peak 1', 'Peak 2', 'Peak 3', 'Peak 4', 'Peak 5', 'Peak 6', 'Amplitude', 'Relative amplitude 2', 'Relative amplitude 3', 'Relative amplitude 4', 'Relative amplitude 5', 'Relative amplitude 6', 'HWHM 1', 'HWHM 2', 'HWHM 3', 'HWHM 4', 'HWHM 5', 'HWHM 6', 'Offset']
# bounds=([f1-5, f2-5, f3-5, f4-5, f5-5, f6-5, 0,0,0,0,0,0,0,0,0,0,0,0,0], [f1+5, f2+5, f3+5, f4+5, f5+5, f6+5, 1e-4, 5,5,5,5,5,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1000])

#* Four peaks

p0=[f1,f2,f3,f4,A,Arel2,Arel3,Arel4,gamma1,gamma2,gamma3,gamma4,250]
names=['Peak 1', 'Peak 2', 'Peak 3', 'Peak 4', 'Amplitude', 'Relative amplitude 2', 'Relative amplitude 3', 'Relative amplitude 4', 'HWHM 1', 'HWHM 2', 'HWHM 3', 'HWHM 4', 'Offset']
bounds=([f1-5, f2-5, f3-5, f4-5, 0,0,0,0,0,0,0,0,0], [f1+5, f2+5, f3+5, f4+5, 1e-4, 5,5,5,1e-7,1e-7,1e-7,1e-7,1000])

#* Three peaks

# p0=[f1,f2,f3,A,Arel2,Arel3,gamma1,gamma2,gamma3,250]
# names=['Peak 1', 'Peak 2', 'Peak 3', 'Amplitude', 'Relative amplitude 2', 'Relative amplitude 3', 'HWHM 1', 'HWHM 2', 'HWHM 3', 'Offset']
# bounds=([f1-5, f2-5, f3-5, 0,0,0,0,0,0,0], [f1+5, f2+5, f3+5, 1e-4, 5,5,1e-7,1e-7,1e-7,1000])



fit_results=fit(four_peaks, f, s, xtitle='$\lambda$ [nm]', ytitle='Counts', title=dataset_name, guess=p0, nombre_params=names, msize=5, legend=False)

