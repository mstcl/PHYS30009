#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of RK4 method and comparison with Euler and midpoint methods;
Object falling under gravity with drag.
Created on Tue Mar  1 17:21:54 2022
@author: phsh
"""
import numpy as np
import matplotlib.pyplot as plt

def f1(v):
    """
    First function definition for Eq 12.3 - somewhat trivial.
    INPUT: v - velocity, m/s (floating point)
    RETURNS: dx/dt = v, m/s (floating point)
    """
    return v

def f2(v,g,a):
    """
    Second function definition for Eq 12.3.
    INPUT: v - velocity, m/s (floating point)
           g - acceleration due to gravity, m/s^2 (floating point)
           a - mass over drag coefficient (m/k), 1/m (floating point)
    RETURNS: dv/dt = g - av|v| (floating point)
    """
    return g-a*v*abs(v)

#
# Define initial conditions, parameters and arrays to hold values
#
g = - 9.81 # m/s^2 acceleration due to gravity
a = 0.005 # 1/m mass over drag coefficient (m/k)
x0 = 1000 # m
v0 = 0 # m/s
t0 = 0 # seconds
tmax = 25 # seconds - maximum time
numpoints = 100 # number of points to consider
t = np.linspace(t0,tmax,numpoints) # array of time values
dt = (tmax-t0)/(numpoints-1) # time increment
x = np.zeros(numpoints) # array to hold x values for RK4
v = np.zeros(numpoints) # array to hold v values for RK4
xe = np.zeros(numpoints) # array to hold x values for Euler
ve = np.zeros(numpoints) # array to hold v values for Euler
xm = np.zeros(numpoints) # array to hold x values for midpoint
vm = np.zeros(numpoints) # array to hold v values for midpoint
x[0] = x0
v[0] = v0
xe[0] = x0
ve[0] = v0
xm[0] = x0
vm[0] = v0
i = 0 # loop counter

while t[i] < tmax and x[i] > 0.0: # stop when out of time or below ground
#
# Runge Kutta 4th order
#
    k1x = f1(v[i])
    k1v = f2(v[i],g,a)
    
    k2x = f1(v[i]+dt*k1v/2)
    k2v = f2(v[i]+dt*k1v/2,g,a)
    
    k3x = f1(v[i]+dt*k2v/2)
    k3v = f2(v[i]+dt*k2v/2,g,a)

    k4x = f1(v[i]+dt*k3v)
    k4v = f2(v[i]+dt*k3v,g,a)

    x[i+1] = x[i] + (dt/6)*(k1x + 2*k2x + 2*k3x + k4x)
    v[i+1] = v[i] + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
#
# Euler method
#
    k1x = f1(ve[i])
    k1v = f2(ve[i],g,a)

    xe[i+1] = xe[i] + dt*k1x
    ve[i+1] = ve[i] + dt*k1v
#
# Modified Euler (midpoint) method
#
    k1x = f1(vm[i])
    k1v = f2(vm[i],g,a)
    
    k2x = f1(vm[i]+dt*k1v/2)
    k2v = f2(vm[i]+dt*k1v/2,g,a)

    xm[i+1] = xm[i] + dt*k2x
    vm[i+1] = vm[i] + dt*k2v

    i += 1
#
# Calculate analytical values
#
xa = x0 - (1/a)*np.log(np.cosh(np.sqrt(-a*g)*t))
#
# Make plots
#
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,5))
fig.suptitle ('Comparing methods for Newtonian freefall with drag, x0={:}m, v0={:}m/s, using {:d} points'.format(x0,v0,numpoints))
ax1.set (xlabel = 'Time (s)' ,ylabel= 'Height (m)' , title = 'Altitude versus time')
ax1.plot (t,x)
ax1.plot (t,xe)
ax1.plot (t,xm)
ax1.plot (t,xa)
ax1.legend(['RK4', 'Euler','Midpoint','Analytical'])
#plt.show()

ax2.set (xlabel = 'Time (s)' ,ylabel= 'Absolute Error in Height (m)' ,title = 'Error compared to analytical solution' )
ax2.plot(t,np.abs(x-xa))
ax2.plot(t,np.abs(xe-xa))
ax2.plot(t,np.abs(xm-xa))
ax2.set_yscale("log")
ax2.legend(['RK4', 'Euler','Midpoint','Analytical'])
#plt.show()
