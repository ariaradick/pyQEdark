"""
Astrophysical parameters associated with various velocity distribution functions
"""

import numpy as np

vEfid = 232 # km/s, simple velocity of earth
vEmin = 215 # km/s
vEmax = 245 # km/s

v0fid = 220 # km/s, velocity dispersion
v0new = 228.6 # km/s
v0min = 200 # km/s
v0max = 280 # km/s

vescfid = 544 # km/s, galactic escape velocity
vescnew = 528 # km/s
vescmin = 450 # km/s
vescmax = 600 # km/s

# Tsallis
qfid = 0.773 # q parameter of Tsallis model
qnew = 1 - (v0new/vescnew)**2 # if we let q depend on the measured params
qmin = 0.5
qmax = 1.3
tsav0fid = 267.2 # km/s, this is from a simulation

# MSW
pfid = 1.5 # p parameter of MSW model
pmin = 0
pmax = 3.0

# debris flow
vflowfid = 340 # km/s
vflowmin = 300
vflowmax = 500

# Less simple parameters:
v_sun = np.array([11.1, 247.24, 7.25]) # km/s
vE_solar = 29.79 # km/s
epsilon_1 = np.array([0.9940, 0.1095, 0.0031])
epsilon_2 = np.array([-0.0517, 0.4945, -0.8677])

omega = 2*np.pi/365.25 # days^-1
t_Mar21 = 79.26 # days

# note that time is in days for vE(t)
vE_fn_solar = lambda t: vE_solar * ( epsilon_1*np.cos(omega*(t-t_Mar21)) + \
                                     epsilon_2*np.sin(omega*(t-t_Mar21)) )

v_lab_max = np.sqrt( np.vdot(vE_fn_solar(t_Mar21)+v_sun,
                             vE_fn_solar(t_Mar21)+v_sun) )
