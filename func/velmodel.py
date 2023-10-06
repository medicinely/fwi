"""
Functions to generate different velocity fields/models
"""

import numpy as np

def readbin(flnam,nz,nx):
	# Read binary file (32 bits)
	with open(flnam,"rb") as fl:
			im = np.fromfile(fl, dtype=np.float32)
	im = im.reshape(nz,nx,order='F')
	return im


def check_model(v, vmin=2000, vmax=3000, disp=False):
    # v[np.where(v > vmax)] = vmax
    # v[np.where(v < vmin)] = vmin
    velmin, velmax = np.min(v), np.max(v)
    # Check if the values are between vmin and vmax
    velmin = np.min(v)
    velmax = np.max(v)
    if (velmin < vmin):
        print("Velocity too small", velmin, vmin)
        raise
    if (velmax > vmax):
        print("Velocity too large", velmax, vmax)
        raise
    if disp==True:
        print("Min. vel:", velmin)
        print("Max. vel:", velmax)    
        print("-- MODEL CHECKED --")