import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def lorentzian(x, x0, gamma):
    return 1 / np.pi * gamma / ((x - x0)**2 + gamma**2) # gamma is the HWHM

def three_peaks_fit(f,f1,f2,f3,A,Arel2,Arel3, gamma1, gamma2, gamma3,offset):
    return A * (lorentzian(f, f1, gamma1) + Arel2 * lorentzian(f, f2, gamma2) + Arel3 * lorentzian(f, f3, gamma3) ) + offset

plt.figure()
f1=960e-9
f2=970e-9
f3=990e-9
A=30e-7
Arel2=0.5
Arel3=3
gamma1=10e-9
gamma2=10e-9
gamma3=15e-9

f_list=np.linspace(930e-9,1010e-9,300)

plt.plot(f_list, three_peaks_fit(f_list,f1,f2,f3,A,Arel2,Arel3, gamma1, gamma2, gamma3,0))

plt.show()

data=np.loadtxt('24_11_18/100_seconds_80_deg_cooling_300_mA_illumination_low_pressure.asc', skiprows=32, delimiter=',')
f_raw=data[:,0]*1e-9
a=np.where(f_raw>940e-9)[0][0]
b=np.where(f_raw<1000e-9)[0][-1]
f=f_raw[a:b]
s=data[a:b,1]

popt, pcov = curve_fit(three_peaks_fit, f, s, p0=[f1,f2,f3,A,Arel2,Arel3, gamma1, gamma2, gamma3,0])
print(popt)
print(np.sqrt(np.diag(pcov)))
plt.figure()
plt.plot(f, s)
plt.plot(f, three_peaks_fit(f,*popt))
plt.show()

print(three_peaks_fit(popt[2],*popt)/three_peaks_fit(popt[0],*popt))


def four_peaks(f, f1, f2, f3, f4, A, Arel2, Arel3, Arel4, gamma1, gamma2, gamma3, gamma4, offset):
    return A * (lorentzian(f, f1, gamma1) + Arel2 * lorentzian(f, f2, gamma2) + Arel3 * lorentzian(f, f3, gamma3) + Arel4 * lorentzian(f, f4, gamma4)) + offset

plt.figure()
f1=960e-9
f2=971e-9
f3=990e-9
f4=980e-9
A=30e-7
Arel2=0.5
Arel3=3
Arel4=0.5

gamma1=10e-9
gamma2=10e-9
gamma3=15e-9
gamma4=10e-9

popt, pcov = curve_fit(four_peaks, f, s, p0=[f1,f2,f3,f4,A,Arel2,Arel3,Arel4, gamma1, gamma2, gamma3, gamma4,0])
print(popt)
print(np.sqrt(np.diag(pcov)))
plt.figure()
plt.plot(f, s)
plt.plot(f, four_peaks(f,*popt))
plt.show()



def six_peaks(f, f1, f2, f3, f4, f5, f6, A, Arel2, Arel3, Arel4, Arel5, Arel6, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, offset):
    return A * (lorentzian(f, f1, gamma1) + Arel2 * lorentzian(f, f2, gamma2) + Arel3 * lorentzian(f, f3, gamma3) + Arel4 * lorentzian(f, f4, gamma4) + Arel5 * lorentzian(f, f5, gamma5) + Arel6 * lorentzian(f, f6, gamma6)) + offset


f1=960e-9
f2=971e-9
f3=981e-9
f4=983e-9
f5=993e-9
f6=995e-9

A=30e-7
Arel2=0.5
Arel3=3
Arel4=0.5
Arel5=3
Arel6=0.5

gamma1=10e-9
gamma2=10e-9
gamma3=15e-9
gamma4=10e-9
gamma5=15e-9
gamma6=10e-9

popt, pcov = curve_fit(six_peaks, f, s, p0=[f1,f2,f3,f4,f5,f6,A,Arel2,Arel3,Arel4,Arel5,Arel6, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6,0])

print(popt)
print(np.sqrt(np.diag(pcov)))
plt.figure()
plt.plot(f, s)
plt.plot(f, six_peaks(f,*popt))
plt.show()