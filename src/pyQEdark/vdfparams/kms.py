"""
Astrophysical parameters associated with various velocity distribution functions
"""

vEfid = 232 # km/s, velocity of earth
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
