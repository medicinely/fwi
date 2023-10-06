"""
Functions to generate different velocity fields/models
"""

import scipy as sc
from scipy.sparse import *
import numpy as np
import matplotlib.pyplot as plt

import gstlearn as gl
from sksparse.cholmod import cholesky

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


def gaussian2d(ranges = [5,10], variance = 10, param = 1, nx = [100,100],mu=2500):
    #Creation of the covariance model (Whittle-Matérn prior)
    model = gl.Model.createFromParam(gl.ECov.BESSEL_K,param = param,
                                     sill = variance, ranges = ranges)
    
    #Creation of the grid 
    mesh = gl.MeshETurbo(nx)
    
    #Creation of the precision matrix (inverse of the covariance)
    precisionOpMat = gl.PrecisionOpCs(mesh,model,0,gl.EPowerPT.ONE,False)
    Qtr = gl.csToTriplet(precisionOpMat.getQ())
    Qmat = sc.sparse.csc_matrix((np.array(Qtr.values), (np.array(Qtr.rows), np.array(Qtr.cols))))
    
    ind = np.arange(Qmat.shape[0])
    ind =np.reshape(ind,(nx[1],nx[0])).T.flatten()
    Qmat = Qmat[ind,:][:,ind]

    #Cholesky decomposition
    cholQ = cholesky(Qmat)
    
    #Random vector
    u = np.random.normal(size=np.prod(Qmat.shape[0]))
    
    # Apply the inverse of the Cholesky decomposition
    
    simu = cholQ.apply_Pt(cholQ.solve_DLt(u))
    
    #Return the simulation (reshaped)
    
    return simu.reshape(mesh.getGrid().getNXs())+mu, Qmat