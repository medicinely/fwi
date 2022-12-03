"""
Functions for objective functions to calculate
the gradiant and residual J
"""

from fwi.func.propagation import prop2d
import numpy as np

def second_order_derivative_old(p, at, az, ax):
	"""
	Calculate second order derivative for p(z,x,t)
	method: finite-difference
	"""
	# extend model to tmin=0 tmax=0 -- insert zeros in dimension t
	nz, nx = len(az), len(ax)
	dt = at[1] - at[0]

	p_extended = np.insert(p,0,np.zeros((nz,nx)), axis=2)
	p_extended = np.append(p_extended,np.zeros((nz,nx,1)), axis=2)
	p_dt_dt = (p_extended[:,:,:-2] - 2 * p_extended[:,:,1:-1] + \
						p_extended[:,:,2:]) / dt**2
	return p_dt_dt

def second_order_derivative(p, at, az, ax):
	"""
	Calculate second order derivative for p(z,x,t)
	method: finite-difference
	"""
	# extend model to tmin=0 tmax=0 -- insert zeros in dimension t
	nz, nx = len(az), len(ax)
	dt = at[1] - at[0]
	p_bound = np.copy(p)
	p_bound[:,:,0] = np.zeros((nz,nx))
	p_bound[:,:,-1] = np.zeros((nz,nx))
	# p_extended = np.append(p_extended,np.zeros((nz,nx,1)), axis=2)
	p_dt_dt = (p_bound[:,:,:-2] - 2 * p_bound[:,:,1:-1] + \
			  p_bound[:,:,2:]) / dt**2
	p_dt_dt = np.insert(p_dt_dt,0,np.zeros((nz,nx)), axis=2)
	p_dt_dt = np.append(p_dt_dt,np.zeros((nz,nx,1)), axis=2)

	return p_dt_dt

def J(vel,d_obs,wsrc,zxsrc,zxrec,at,az,ax,next,device):
	nz, nx = len(az), len(ax)
	if vel.ndim == 1: vel = np.reshape(vel.flatten(),(nz,nx))
	p_fwd = prop2d(wsrc,zxsrc,vel,at,az,ax,next,device)
	d = p_fwd[zxrec[0], zxrec[1], :]
	residual = d - d_obs
	J = 0.5 * np.sum(residual ** 2) # Residual is the half of the L2 norm square
	print("J = %.5f" % J)
	return J

def gradiant(vel,d_obs,wsrc,zxsrc,zxrec,at,az,ax,next,device):
	"""
	Calculate gradiant using adjoint state method
	"""
	# Calculate forward propagated p_fwd(z,x,t) (to update)
	nz, nx, nt = len(az), len(ax), len(at)
	if vel.ndim == 1: vel = np.reshape(vel,(nz,nx))
	p_fwd = prop2d(wsrc,zxsrc,vel,at,az,ax,next, device)
	receiver_depth = zxrec[0]
	d = p_fwd[zxrec[0], zxrec[1], :]
	residual = d - d_obs # residual shape (n_rec, nt) - residual.shape=(5,801)
	# Calculate back propagated p_back(z,x,t)
	p_back = prop2d(np.flip(residual,axis=1), zxrec, vel, at, az, ax, next, device)
	p_back = np.flip(p_back,axis=2)
	# Calculate second order time derivative of p_fwd(z,x,t)
	p_dt_dt = second_order_derivative(p_fwd, at, az, ax)
	# Calculate gradiant
	G = 2/vel**3  * np.sum(p_back * p_dt_dt, axis=2) # G.shape = (201,201)

	return G.flatten()
