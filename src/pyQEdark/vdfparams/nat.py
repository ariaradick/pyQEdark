from pyQEdark.constants import ckms

vEfid = 232/ckms # (km/s), velocity of earth
vEmin = 215/ckms # (km/s)
vEmax = 245/ckms # (km/s)

v0fid = 220/ckms # (km/s), velocity dispersion
v0new = 228.6/ckms # (km/s)
v0min = 200/ckms # (km/s)
v0max = 280/ckms # (km/s)

vescfid = 544/ckms # (km/s), galactic escape velocity
vescnew = 528/ckms # (km/s)
vescmin = 450/ckms # (km/s)
vescmax = 600/ckms # (km/s)

# Tsallis
qfid = 0.773 # q parameter of Tsallis model
qnew = 1 - (v0new/vescnew)**2 # if we let q depend on the measured params
qmin = 0.5
qmax = 1.3
tsav0fid = 267.2/ckms # (km/s), this is from a simulation

# MSW
pfid = 1.5 # p parameter of MSW model
pmin = 0
pmax = 3.0

# debris flow
vflowfid = 340/ckms # (km/s)
vflowmin = 300/ckms
vflowmax = 500/ckms

del ckms
