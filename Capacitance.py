# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:10:11 2022

@author: Olivia Keene
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as optimize
import math
from numpy import log as ln
#%% experiment 1 plots
time1,pd1=np.loadtxt('Python Files/Data/Capacitance experiment data/CAPPP15.csv',skiprows=2, delimiter=',',unpack=True)
plt.plot(time1,pd1,'b.')
plt.grid()
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("potential difference (V)", fontsize=14) #labels y-axis
plt.title("Charging capacitor plot", fontsize=16)
plt.savefig("Python Files/Output/Exp1_Charge.jpg")
plt.show()
time2,pd2=np.loadtxt('Python Files/Data/Capacitance experiment data/CAPPP16.csv',skiprows=2, delimiter=',',unpack=True)
plt.plot(time2,pd2,'m,')
plt.grid()
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("potential difference (V)", fontsize=14) #labels y-axis
plt.title("Discharging capacitor plot", fontsize=16)
plt.savefig("Python Files/Output/Exp1_Discharge.jpg")
plt.show()
#%% experiment 2 plots
time3,pd31,pd32=np.loadtxt('Python Files/Data/Capacitance experiment data/CAPPP12.csv',skiprows=2, delimiter=',',unpack=True)
plt.plot(time3,pd31,'b-')
plt.plot(time3,pd32,'m-')
plt.grid()
plt.xlim(0,0.0006)
plt.ylim(-1.5,1.5)
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("potential difference (V)", fontsize=14) #labels y-axis
plt.title("Square wave 5kHz input with small capacitor plot", fontsize=15)
plt.savefig("Python Files/Output/Exp2_5kHz.jpg")
#plt.legend(['input voltage','voltage across capacitor'],loc=1)
plt.show()
time4,pd41,pd42=np.loadtxt('Python Files/Data/Capacitance experiment data/CAPPP09.csv',skiprows=2, delimiter=',',unpack=True)
plt.plot(time4,pd41,'b-')
plt.plot(time4,pd42,'m-')
plt.grid()
#plt.xlim(0,0.0006)
plt.ylim(-1.5,2.0)
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("potential difference (V)", fontsize=14) #labels y-axis
plt.title("Square wave 100Hz input with small capacitor plot", fontsize=15)
plt.savefig("Python Files/Output/Exp2_100Hz.jpg")
plt.legend(['input voltage','voltage across capacitor'],loc=1)
plt.show()
time5,pd51,pd52=np.loadtxt('Python Files/Data/Capacitance experiment data/CAPPP10.csv',skiprows=2, delimiter=',',unpack=True)
plt.plot(time5,pd51,'b-')
plt.plot(time5,pd52,'m-')
plt.grid()
#plt.xlim(0,0.0006)
plt.ylim(-1.5,2.0)
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("potential difference (V)", fontsize=14) #labels y-axis
plt.title("Square wave 500Hz input with small capacitor plot", fontsize=15)
plt.savefig("Python Files/Output/Exp2_500Hz.jpg")
plt.legend(['input voltage','voltage across capacitor'],loc=1)
plt.show()
time6,pd61,pd62=np.loadtxt('Python Files/Data/Capacitance experiment data/CAPPP11.csv',skiprows=2, delimiter=',',unpack=True)
plt.plot(time6,pd61,'b-')
plt.plot(time6,pd62,'m-')
plt.grid()
#plt.xlim(0,0.0006)
plt.ylim(-1.5,2.0)
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("potential difference (V)", fontsize=14) #labels y-axis
plt.title("Square wave 1kHz input with small capacitor plot", fontsize=15)
plt.savefig("Python Files/Output/Exp2_1kHz.jpg")
plt.legend(['input voltage','voltage across capacitor'],loc=1)
plt.show()
time7,pd71,pd72=np.loadtxt('Python Files/Data/Capacitance experiment data/CAPPP13.csv',skiprows=2, delimiter=',',unpack=True)
plt.plot(time7,pd71,'b-')
plt.plot(time7,pd72,'m-')
plt.grid()
#plt.xlim(0,0.0006)
plt.ylim(-1.5,2.0)
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("potential difference (V)", fontsize=14) #labels y-axis
plt.title("Square wave 10kHz input with small capacitor plot", fontsize=15)
plt.savefig("Python Files/Output/Exp2_10kHz.jpg")
plt.legend(['input voltage','voltage across capacitor'],loc=1)
plt.show()
time8,pd81,pd82=np.loadtxt('Python Files/Data/Capacitance experiment data/CAPPP14.csv',skiprows=2, delimiter=',',unpack=True)
plt.plot(time8,pd81,'b-')
plt.plot(time8,pd82,'m-')
plt.grid()
#plt.xlim(0,0.0006)
plt.ylim(-1.5,2.0)
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("potential difference (V)", fontsize=14) #labels y-axis
plt.title("Square wave 15kHz input with small capacitor plot", fontsize=15)
plt.savefig("Python Files/Output/Exp2_15kHz.jpg")
plt.legend(['input voltage','voltage across capacitor'],loc=1)
plt.show()
#%% experiment 3
time9,pd91,pd92=np.loadtxt('Data/Capacitance experiment data/CAPPP17.csv',skiprows=2, delimiter=',',unpack=True)
plt.plot(time9,pd91,'b-')
plt.plot(time9,pd92,'m-')
plt.grid()
#plt.plot(time9,-np.sin(100000*np.pi*time9),'g-')
#plt.xlim(0,0.0006)
plt.ylim(-1.5,1.5)
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("potential difference (V)", fontsize=14) #labels y-axis
plt.title("50kHz Sine wave input with small capacitor plot", fontsize=15)
#plt.savefig("Python Files/Output/Exp2_15kHz.jpg")



def func(x,A,p):
    sine = -A*np.sin(100000*np.pi*x+p)
    return sine

A_guess=(0.6)
p_guess=(0)

initial_guess=np.array([A_guess,p_guess]) #array of initial guess values containing, in order: estimate amplitude, estimate mean wavelength, estimate sigma, estimate linear gradient, estimate linear y-intercept
fit,fit_cov=sp.optimize.curve_fit(func,time9,pd92,initial_guess, maxfev=1000000) 
x=np.linspace(-6e-5,6e-5,10000)
#plt.plot(x,func(x,fit[0],fit[1]),'r')
a=-fit[1]
print('Amplitude is ',fit[0],' +/- ',cov[0,0],' V and phase difference is ',a,' +/- ', cov[1,1],' rad')

Ct=1*np.sin(a)/(2*np.pi*50000*fit[0]*6800)
C3=Ct-20e-12
C3_u=C3*np.sqrt((4.4294e-3/1)**2+(np.sqrt(np.cos(a)*(fit_cov[1,1])**2)/np.sin(a))**2+(fit_cov[0,0]/fit[0])**2+0.05**2)
print('total capacitance is ',Ct,' +/- ',C3_u,' F, and small capacitor capacitance is ', C3,' +/- ',C3_u,' F')
plt.legend(['input voltage','voltage across capacitor','optimised function'],loc=1)
plt.show()

#%% experiment 1 data analysis

plt.plot(time2[19000:126000],np.log(pd2[19000:126000]),'r')
fit,cov = np.polyfit(time2[19000:126000],np.log(pd2[19000:126000]),1,w=None,cov=True)
line=np.poly1d(fit)
t=np.linspace(7.77,31.27,100000)
plt.plot(t,line(t),'b-')
plt.grid()
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("natural log of potential difference", fontsize=14) #labels y-axis
plt.title("Large capacitor logarithmic discharge plot", fontsize=16)
#plt.savefig("Python Files/Output/Exp1_Charge.jpg")
plt.show()
v0=np.sum(pd2[0:18000])/18000
v0_u=np.std(pd2[0:18000])
print(v0,v0_u, fit[0],np.sqrt(cov[0,0]))
C=-1/(10000*fit[0])
C_u=C*np.sqrt(0.05**2+(cov[0,0]/fit[0])**2)
print("capacitance of large capacitor is ",C," +/- ",C_u," F")

#%% experiment 2 data analysis
#plt.plot(time3[45286:67012],np.log(pd32[45286:67012]),'r')
plt.plot(time3[47386:67012],np.log(pd32[47386:67012]+0.7),'r')
fit2,cov2 = np.polyfit(time3[47386:67012],np.log(pd32[47386:67012]+0.7),1,w=None,cov=True)
line=np.poly1d(fit2)
t2=np.linspace(0.000205708, 0.0003,100000)
plt.plot(t2,line(t2),'b-')
plt.grid()
plt.xticks([0.0002100, 0.00023,0.00025,0.00027,0.00029,0.00031])
plt.xlabel("time (s)", fontsize=14) #labels x-axis
plt.ylabel("natural log of potential difference", fontsize=14) #labels y-axis
plt.title("Small capacitor logarithmic discharge plot", fontsize=16)
plt.savefig("Python Files/Output/Exp1_Charge.jpg")
plt.show()
v0=np.sum(pd32[47350:47386])/36
print(v0)
C2=-1/(10000*fit2[0])
C_u2=C2*np.sqrt((500/10000)**2+(cov2[0,0]/fit2[0])**2)
print("capacitance of small capacitor is ",C2," +/- ",C_u2," F")

