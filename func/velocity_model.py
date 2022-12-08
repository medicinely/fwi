"""
Functions to generate different velocity fields/models
"""

import scipy as sc
from scipy.sparse import *
import numpy as np
import matplotlib.pyplot as plt


def check_model(v, vmin=2000, vmax=3000, disp=False):
    v[np.where(v > vmax)] = vmax
    v[np.where(v < vmin)] = vmin
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


def gaussian2d(ranges=[5, 10], variance=1, param=1, nx=[201, 201], mean=0):
    import gstlearn as gl
    from sksparse.cholmod import cholesky   
    #Creation of the covariance model (Whittle-Matérn prior)
    model = gl.Model.createFromParam(gl.ECov.BESSEL_K,
                                     param=param,
                                     sill=variance,
                                     ranges=ranges)

    #Creation of the grid
    mesh = gl.MeshETurbo(nx)

    #Creation of the precision matrix (inverse of the covariance)
    precisionOpMat = gl.PrecisionOpCs(mesh, model, 0, gl.EPowerPT.ONE, False)
    Qtr = gl.csToTriplet(precisionOpMat.getQ())
    Qmat = sc.sparse.csc_matrix(
        (np.array(Qtr.values), (np.array(Qtr.rows), np.array(Qtr.cols))))

    #Cholesky decomposition
    cholQ = cholesky(Qmat)

    #Random vector
    u = np.random.normal(size=np.prod(Qmat.shape[0]))

    # Apply the inverse of the Cholesky decomposition

    simu = cholQ.apply_Pt(cholQ.solve_DLt(u))

    #Return the simulation (reshaped)

    return simu.reshape(mesh.getGrid().getNXs()) + mean