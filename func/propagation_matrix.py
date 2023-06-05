"""
Functions for 2D wave propagation
"""
import numpy as np
import scipy.sparse as sp
# import cupy as cp
from math import pi, sqrt, exp, sin, cos

precision = np.float32


def defmodel(vmin, vmax, fmax, nz, nx, nt, izsrc=[100], ixsrc=[10],ext=100):
	"""
	Initialize the model
	Input: vmin, vmax, fmax, nz, nx, nt
	Output: axis az, ax, at
	"""
	# Key parameters
	vmin, vmax, fmax = 2000.,3000.,25 
	# Deduce dzmax and dtmax (in 1D)
	dzmax, dxmax, dtmax = param2grid(vmin, vmax, fmax)
	dz = dzmax
	dx = dxmax
	dt = dtmax
	print("dz,dx,dt (m):", dz,dx,dt)
	#-----------------------------------------------
	# Definition of the source wavelet (Ricker function)
	# aw is the time axis
	wsrc, aw = defsrc(fmax, dt)
	# User input
	# Define the size of the model and the number of time samples
	print("Model dimension [nz,nx,nt]: ", nz, nx, nt)
	# Define the x and time axis
	fz,fx,ft = 0.,0.,aw[0]  # First time given by the source wavelet
	az = fz + dz * np.arange(nz)
	ax = fx + dx * np.arange(nx)
	at = ft + dt * np.arange(nt)

	# Define the source location
	zxsrc = np.array([izsrc, ixsrc])

	return az, ax, at, ext, wsrc, zxsrc


def param2grid(vmin, vmax, fmax):
	"""
	Definition of the grid in space and time
	"""
	# Dispersion condition (10 points per wavelength)
	dzmax = vmin / fmax / 10.
	dxmax = vmin / fmax / 10.
	# Stabiblity condition
	# Factor 0.8 in the case of PML
	dtmax = dzmax / vmax / sqrt(2.) * 0.90
	return dzmax, dxmax, dtmax


def defsrc(fmax, dt):
	"""
	Definition of the source (Ricker) function
	Ricker wavelet with central frequency fmax/2.5
	Ricker = 2nd-order derivative of a Gaussian function
	Advantage: the frequency content of that shape is controlled by the fmax argument
	"""
	fc   = fmax / 2.5    # Central frequency
	ns2  = int(2/fc/dt)
	ns   = 1 + 2*ns2     # Size of the source
	wsrc = np.zeros(ns)
	aw   = np.zeros(ns)  # Time axis
	for it in range(ns):
		a1 = float(it-ns2)*fc*dt*pi
		a2 = a1**2
		wsrc[it] = (1-2*a2)*exp(-a2)
		aw[it]   = float(it-ns2)*dt
	return wsrc, aw

def extend_model(vel,next):
	"""
	Extension of the model (to limit the edge effects)
	"""
	nz   = np.shape(vel)[0]
	nx   = np.shape(vel)[1]
	nze  = nz + 2*next
	nxe  = nx + 2*next
	vele = np.zeros([nze,nxe], dtype=precision)
	# Central part
	vele[next:nze-next,next:nxe-next] = vel
	# Top and bottomB
	for ix in range(next,nxe-next):
		for iz in range(next):
			vele[iz,ix]       = vel[0,ix-next]
			vele[nze-1-iz,ix] = vel[nz-1,ix-next]
	# Left and right
	for ix in range(next):
		for iz in range(next,nze-next):
			vele[iz,ix]       = vel[iz-next,0]
			vele[iz,nxe-1-ix] = vel[iz-next,nx-1]
	# Corners
	for ix in range(next):
		for iz in range(next):
			vele[iz,ix]             = vel[0,0]
			vele[nze-1-iz,ix]       = vel[nz-1,0]
			vele[iz,nxe-1-ix]       = vel[0,nx-1]
			vele[nze-1-iz,nxe-1-ix] = vel[nz-1,nx-1]
	return vele


def prop2d(wsrc, zxsrc, zxrec, vel, at, az, ax, next, device='cpu'):
	"""
	2d wave propagation with multiple sources
	"""

	if device=='cpu':
		"""
		2d wave propagation
		Resolution with finite differences
		Orders 2 in time and space
		with absorbing boundaries (Clayton and Engquist)
		Vectorial implementation (much faster)
		"""
		nabs  = 10
		next2 = nabs + next
		nt    = len(at)
		nz    = len(az)
		nx    = len(ax)
		dz    = az[1] - az[0]
		dx    = ax[1] - ax[0]
		dt    = at[1] - at[0]   
		_dz2   = 1./dz**2
		_dx2   = 1./dx**2

		# Calculate source waveform
		wsrc = np.array([wsrc], dtype=precision) if wsrc.ndim == 1 else np.array(wsrc, dtype=precision) # convert source wavelet from 1d to 2d
		zxsrc = np.array(zxsrc) # source location to array
		pwsrc = np.zeros([nz, nx, nt]) # creat a initial p with zeros
		pwsrc[zxsrc[0,:], zxsrc[1,:], :wsrc.shape[1]] = wsrc # insert source wavelet

		# Extend the model
		nze  = nz + 2*next2
		nxe  = nx + 2*next2
		vele = extend_model(vel,next2)

		# Shift the source by next
		asrc = np.zeros([nze-2*nabs-2,nxe-2*nabs-2], dtype=precision)
		pm    = np.zeros([nze,nxe], dtype=precision) # Previous wave field
		pp    = np.zeros([nze,nxe], dtype=precision)
		fact = (dt * vele[nabs:-nabs,nabs:-nabs])**2

		# Construct the sparse Laplacian matrix
		nze = nz + 2 * next2
		nxe = nx + 2 * next2

		# Calculate the size of the Laplacian matrix
		nzz, nxx = nz + 2*next, nx + 2*next
		size = nzz * nxx

		# Construct the Laplacian matrix
		diagonal = -2*np.ones(size)*(_dz2+_dx2)
		off_diagonal_x = np.ones(size - 1) * _dz2
		off_diagonal_z = np.ones(size - nxx) * _dx2

		# Set the off-diagonal elements of the Laplacian matrix for the x- and z-direction neighbors
		off_diagonal_x[nxx - 1::nxx] = 0
		off_diagonal_z[-nxx:] = 0

		# Create the Laplacian matrix L using the diagonal and off-diagonal values
		L = sp.diags([off_diagonal_x, off_diagonal_z, diagonal, off_diagonal_z, off_diagonal_x],
								[-1, -nxx, 0, nxx, 1], shape=(size, size))

		# Create the Transform matrix T using wave equation
		T = sp.diags(fact.flatten(), 0, format='csr').dot(L) + sp.diags(2*np.ones(size), 0, format='csr')

		d_obs = [np.zeros(zxrec.shape[1])]
		p = [np.zeros((nz,nx))]

		for it in range(1,nt-1): # From 1 to nt-1
			pt = pp.copy()
			# ptwsrc = np.zeros((nz,nx))
			# ptwsrc[zxsrc[0,:], zxsrc[1,:]] = extend_wsrc[it]
			# asrc[next:-next,next:-next] = ptwsrc[1:-1,1:-1]      
			asrc[next:-next,next:-next] = pwsrc[1:-1,1:-1,it]		#	0.0002s
			pp[1+nabs:-1-nabs,1+nabs:-1-nabs] = \
							T.dot(pt[nabs:-nabs, nabs:-nabs].flatten()).reshape(nzz, nxx)[1:-1, 1:-1]   \
							- pm[1+nabs:-1-nabs,1+nabs:-1-nabs]\
							+ asrc * fact[1:-1, 1:-1]   
			pm = pt

			# One-way equation (bottom part)
			pp[nze-1-nabs:nze,:nxe] = pt[nze-1-nabs:nze,:nxe] - \
							vele[nze-1-nabs:nze,:nxe]*dt/dz* \
							(pt[nze-1-nabs:nze,:nxe]-pt[nze-2-nabs:nze-1,:nxe])
			# One-way equation (top part)
			pp[:1+nabs,:nxe] = pt[:1+nabs,:nxe] + \
							vele[:1+nabs,:nxe]*dt/dz* \
							(pt[1:2+nabs,:nxe]-pt[:1+nabs,:nxe])
			# One-way equation (right part)
			pp[:nze,nxe-1-nabs:nxe] = pt[:nze,nxe-1-nabs:nxe] - \
							vele[:nze,nxe-1-nabs:nxe]*dt/dx* \
			(pt[:nze,nxe-1-nabs:nxe] - pt[:nze,nxe-2-nabs:nxe-1])
			# One-way equation (left part)
			pp[:nze,:1+nabs] = pt[:nze,:1+nabs] + \
							vele[:nze,:1+nabs]*dt/dx* \
							(pt[:nze,1:2+nabs]-pt[:nze,:1+nabs])

			d_obs.append(pp[next2:nze-next2,next2:nxe-next2][zxrec[0], zxrec[1]])
			p.append(pp[next2:nze-next2,next2:nxe-next2].copy())

		d_obs.append(np.zeros(zxrec.shape[1]))
		d_obs = np.array(d_obs).T
		p.append(np.zeros((nz,nx)))
		p = np.moveaxis(np.array(p), 0, -1)

	return p, d_obs


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
	_, d = prop2d(wsrc,zxsrc,zxrec,vel,at,az,ax,next,device)
	residual = d - d_obs
	J = 0.5 * np.sum(residual ** 2) # Residual is the half of the L2 norm square
	print("J =", J)

	return J


def gradiant(vel,d_obs,wsrc,zxsrc,zxrec,at,az,ax,next,device):
	"""
	Calculate gradiant using adjoint state method
	"""
	# Calculate forward propagated p_fwd(z,x,t) (to update)
	nz, nx, nt = len(az), len(ax), len(at)
	if vel.ndim == 1: vel = np.reshape(vel,(nz,nx))
	p_fwd, d = prop2d(wsrc,zxsrc,zxrec,vel,at,az,ax,next,device)
	receiver_depth = zxrec[0]
	residual = d - d_obs # residual shape (n_rec, nt) - residual.shape=(5,801)
	# Calculate back propagated p_back(z,x,t)
	p_back, _ = prop2d(np.flip(residual,axis=1), zxrec, zxsrc, vel, at, az, ax, next, device)
	p_back = np.flip(p_back,axis=2)
	# Calculate second order time derivative of p_fwd(z,x,t)
	p_dt_dt = second_order_derivative(p_fwd, at, az, ax)
	# Calculate gradiant
	G = 2/vel**3  * np.sum(p_back * p_dt_dt, axis=2) # G.shape = (201,201)

	return G.flatten()