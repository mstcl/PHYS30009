#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of RK4 method using scipy.integrate.solve_ivp;
Object falling under gravity with drag.
Created on Tue Mar  1 17:21:54 2022
@author: phsh
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f12(t,state,g,a):
    """
    Function definitions for Eq 12.3 - somewhat trivial.
    INPUT: t - independent variable time (s) (not used)
           state - tuple of dependent variables (x,v)
                   x: position (not required), m (floating point)
                   v: velocity, m/s (floating point)
           g - acceleration due to gravity, m/s^2 (floating point)
           a - mass over drag coefficient (m/k), 1/m (floating point)
    RETURNS: (dx/dt,dv/dt) = (v,g - av|v|) (floating point tuple)
    """
    x,v = state
    dxdt = f1(v)  # this is f1
    dvdt = f2(v,g,a)  # this is f2
    return (dxdt,dvdt)

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
rtol=1e-5 # relative tolerance of solve_ivp method
atol=1e-6 # (m) absolute tolerance of solve_ivp method
t = np.linspace(t0,tmax,numpoints) # array of time values
dt = (tmax-t0)/(numpoints-1) # time increment
x = np.zeros(numpoints) # array to hold x values for RK4
v = np.zeros(numpoints) # array to hold v values for RK4

x[0] = x0
v[0] = v0

i = 0 # loop counter for RK4

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

    i += 1
#
# Calculate analytical values
#
xa = x0 - (1/a)*np.log(np.cosh(np.sqrt(-a*g)*t))
#
# Apply solver
#
result_solve_ivp = solve_ivp(f12, (t0,tmax), (x0,v0), args=(g,a),method='RK45', t_eval=t,rtol=rtol,atol=atol)
#
# Make plots
#
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,5))
fig.suptitle ('Comparing methods for Newtonian freefall with drag, x0={:}m, v0={:}m/s, using {:d} points; rtol={:}, atol={:}m'.format(x0,v0,numpoints,rtol,atol))
ax1.set (xlabel = 'Time (s)' ,ylabel= 'Height (m)' , title = 'Altitude versus time')
ax1.plot (t,x)
ax1.plot(t,result_solve_ivp.y[0,:])
ax1.legend(['Explicit RK4', 'solve_ivp with RK45'])
#plt.show()

ax2.set (xlabel = 'Time (s)' ,ylabel= 'Absolute Error in Height (m)' ,title = 'Error compared to analytical solution' )
ax2.plot(t,np.abs(x-xa))
ax2.plot(t,np.abs(result_solve_ivp.y[0,:]-xa))
ax2.set_yscale("log")
ax2.legend(['Explicit RK4', 'solve_ivp with RK45'])
#plt.show()

