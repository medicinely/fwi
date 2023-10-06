"""
Functions for 2D wave propagation
"""
import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx
from math import pi, sqrt, exp, sin, cos


def defmodel(vmin, vmax, fmax, nz, nx, nt, izsrc=[100], ixsrc=[10],ext=100):
	"""
	Initialize the model
	Input: vmin, vmax, fmax, nz, nx, nt
	Output: axis az, ax, at
	"""
	# Key parameters
	# vmin, vmax, fmax = 2000.,3000.,25 
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
	ns2  = int(1.2/fc/dt)
	ns   = 1 + 2*ns2     # Size of the source
	wsrc = np.zeros(ns)
	aw   = np.zeros(ns)  # Time axis
	for it in range(ns):
		a1 = float(it-ns2)*fc*dt*pi
		a2 = a1**2
		wsrc[it] = (1-2*a2)*exp(-a2)
		aw[it]   = float(it-ns2)*dt
	return wsrc, aw

def extend_model(v,nz,nx,next):
	"""
	Extension of the model (to limit the edge effects)
	"""
	vel = v.reshape((nz,nx))
	nze  = nz + 2*next
	nxe  = nx + 2*next
	vele = np.zeros([nze,nxe])
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
	return vele.flatten()

def prop2d(wsrc, zxsrci, zxrec, vel, at, az, ax, next, device='gpu'):
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

		nze = nz + 2 * next2
		nxe = nx + 2 * next2
		size = nze * nxe

		# Calculate source waveform
		wsrc = np.array([wsrc]) if wsrc.ndim == 1 else np.array(wsrc) # convert source wavelet from 1d to 2d
		zxsrci = np.array(zxsrci) # source location to array

		# Extend the model
		nze  = nz + 2*next2
		nxe  = nx + 2*next2
		vele = extend_model(vel,nz,nx,next2)
		vele = vele.flatten()

		# Center Part (Laplacian)
		mask0 = np.full((nze, nxe), False)
		mask0[1+nabs:-1-nabs,1+nabs:-1-nabs] = True
		mask0 = mask0.flatten()

		# Construct the Laplacian matrix
		fact = (dt * vele)**2

		# Initialize arrays
		diagonal = np.zeros(size)
		off_diagonal_x_left = np.zeros(size)
		off_diagonal_x_right = np.zeros(size)
		off_diagonal_z_left = np.zeros(size)
		off_diagonal_z_right = np.zeros(size)

		# Set values for interior points
		diagonal[mask0] = (2*np.ones(size) + fact*(-2*np.ones(size)*(_dz2+_dx2)))[mask0]
		off_diagonal_x_left[mask0] = (fact*np.ones(size) * _dz2)[mask0]
		off_diagonal_x_right[mask0] = (fact*np.ones(size) * _dz2)[mask0]
		off_diagonal_z_left[mask0] = (fact*np.ones(size) * _dx2)[mask0]
		off_diagonal_z_right[mask0] = (fact*np.ones(size) * _dx2)[mask0]

		# Calculate factors for boundary conditions
		factz = (-dt/dz) * vele
		factx = (-dt/dx) * vele

		# Bottom part
		mask1 = np.full((nze, nxe), False)
		mask1[nze-1-nabs:nze,:nxe] = True
		mask1 = mask1.flatten()
		diagonal[mask1] = 1 + factz[mask1]
		off_diagonal_z_left[mask1] = -factz[mask1]

		# Top part
		mask2 = np.full((nze, nxe), False)
		mask2[:1+nabs,:nxe] = True
		mask2 = mask2.flatten()
		diagonal[mask2] = 1 + factz[mask2]
		off_diagonal_z_right[mask2] = -factz[mask2]

		# Right part
		mask3 = np.full((nze, nxe), False)
		mask3[:nze,nxe-1-nabs:nxe] = True
		mask3 = mask3.flatten()
		diagonal[mask3] = 1 + factx[mask3]
		off_diagonal_x_left[mask3] = -factx[mask3]

		# Left part
		mask4 = np.full((nze, nxe), False)
		mask4[:nze,:1+nabs]  = True
		mask4 = mask4.flatten()
		diagonal[mask4] = 1 + factx[mask4]
		off_diagonal_x_right[mask4] = -factx[mask4]

		# Construct the sparse Laplacian matrix
		A = sp.diags([off_diagonal_x_left[1:], off_diagonal_z_left[nxe:], diagonal, off_diagonal_z_right, off_diagonal_x_right],
								[-1, -nxe, 0, nxe, 1], shape=(size, size), format='csr')

		# Wavefield p mask (initial center part)
		maskp = np.full((nze, nxe), 0)
		maskp[next+nabs:nze-next-nabs,next+nabs:nxe-next-nabs]  = 1
		maskp = maskp.flatten()

		pm = np.zeros(size) # Previous wave field
		pt = np.zeros(size)
		pp = np.zeros(size)

		p = [np.zeros(nz*nx)]
		p.append(np.zeros(nz*nx))

		srcsteps = wsrc.shape[1] # Determine the time steps of the source
		for it in range(1,nt-1): # From 1 to nt-1
			pm = pt.copy()
			pt = pp.copy()
			# pp = A.dot(pt) - pm*mask0 + pwsrc[:,it]*fact*mask0
			if it<srcsteps:
				cen = np.zeros((nz,nx))
				cen[zxsrci[0,:], zxsrci[1,:]] = wsrc[:,it]
				asrcit = np.pad(cen, next2, mode='constant').flatten()
				srcterm = asrcit*fact*mask0
			else: srcterm = np.zeros(size)

			pp = A.dot(pt) - pm*mask0 + srcterm
			p.append(pp[maskp==1])

		p = np.moveaxis(np.array(p), 0, -1)
		d_obs = p.reshape((nz,nx,nt))[zxrec[0], zxrec[1], :]

	elif device=='gpu':
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

		nze = nz + 2 * next2
		nxe = nx + 2 * next2
		size = nze * nxe

		# Calculate source waveform
		wsrc = cp.array([wsrc]) if wsrc.ndim == 1 else cp.array(wsrc) # convert source wavelet from 1d to 2d
		zxsrci = cp.array(zxsrci) # source location to array

		# Extend the model
		nze  = nz + 2*next2
		nxe  = nx + 2*next2
		vele = extend_model(vel,nz,nx,next2)
		vele = cp.array(vele.flatten())

		# Center Part (Laplacian)
		mask0 = cp.full((nze, nxe), False)
		mask0[1+nabs:-1-nabs,1+nabs:-1-nabs] = True
		mask0 = mask0.flatten()

		# Construct the Laplacian matrix
		fact = (dt * vele)**2

		# Initialize arrays
		diagonal = cp.zeros(size)
		off_diagonal_x_left = cp.zeros(size)
		off_diagonal_x_right = cp.zeros(size)
		off_diagonal_z_left = cp.zeros(size)
		off_diagonal_z_right = cp.zeros(size)

		# Set values for interior points
		diagonal[mask0] = (2*cp.ones(size) + fact*(-2*cp.ones(size)*(_dz2+_dx2)))[mask0]
		off_diagonal_x_left[mask0] = (fact*cp.ones(size) * _dz2)[mask0]
		off_diagonal_x_right[mask0] = (fact*cp.ones(size) * _dz2)[mask0]
		off_diagonal_z_left[mask0] = (fact*cp.ones(size) * _dx2)[mask0]
		off_diagonal_z_right[mask0] = (fact*cp.ones(size) * _dx2)[mask0]

		# Calculate factors for boundary conditions
		factz = (-dt/dz) * vele
		factx = (-dt/dx) * vele

		# Bottom part
		mask1 = cp.full((nze, nxe), False)
		mask1[nze-1-nabs:nze,:nxe] = True
		mask1 = mask1.flatten()
		diagonal[mask1] = 1 + factz[mask1]
		off_diagonal_z_left[mask1] = -factz[mask1]

		# Top part
		mask2 = cp.full((nze, nxe), False)
		mask2[:1+nabs,:nxe] = True
		mask2 = mask2.flatten()
		diagonal[mask2] = 1 + factz[mask2]
		off_diagonal_z_right[mask2] = -factz[mask2]

		# Right part
		mask3 = cp.full((nze, nxe), False)
		mask3[:nze,nxe-1-nabs:nxe] = True
		mask3 = mask3.flatten()
		diagonal[mask3] = 1 + factx[mask3]
		off_diagonal_x_left[mask3] = -factx[mask3]

		# Left part
		mask4 = cp.full((nze, nxe), False)
		mask4[:nze,:1+nabs]  = True
		mask4 = mask4.flatten()
		diagonal[mask4] = 1 + factx[mask4]
		off_diagonal_x_right[mask4] = -factx[mask4]

		# Construct the sparse Laplacian matrix
		# A = sp.diags([off_diagonal_x_left[1:], off_diagonal_z_left[nxe:], diagonal, off_diagonal_z_right, off_diagonal_x_right],
		# 						[-1, -nxe, 0, nxe, 1], shape=(size, size), format='csr')
		
		A = cupyx.scipy.sparse.diags([off_diagonal_x_left[1:], off_diagonal_z_left[nxe:], diagonal, off_diagonal_z_right, off_diagonal_x_right],
								[-1, -nxe, 0, nxe, 1], shape=(size, size), format='csr')
		
		# Wavefield p mask (initial center part)
		maskp = cp.full((nze, nxe), 0)
		maskp[next+nabs:nze-next-nabs,next+nabs:nxe-next-nabs]  = 1
		maskp = maskp.flatten()

		pm = cp.zeros(size) # Previous wave field
		pt = cp.zeros(size)
		pp = cp.zeros(size)

		p = [cp.zeros(nz*nx)]
		p.append(cp.zeros(nz*nx))

		srcsteps = wsrc.shape[1] # Determine the time steps of the source
		for it in range(1,nt-1): # From 1 to nt-1
			pm = pt.copy()
			pt = pp.copy()
			# pp = A.dot(pt) - pm*mask0 + pwsrc[:,it]*fact*mask0
			if it<srcsteps:
				cen = cp.zeros((nz,nx))
				cen[zxsrci[0,:], zxsrci[1,:]] = wsrc[:,it]
				asrcit = cp.pad(cen, next2, mode='constant').flatten()
				srcterm = asrcit*fact*mask0
			else: srcterm = cp.zeros(size)

			pp = A.dot(pt) - pm*mask0 + srcterm
			p.append(pp[maskp==1])

		p = cp.moveaxis(cp.array(p), 0, -1)
		d_obs = p.reshape((nz,nx,nt))[zxrec[0], zxrec[1], :]
		p, d_obs = cp.asnumpy(p), cp.asnumpy(d_obs)

	return p, d_obs

def second_order_derivative_cpu(p, at, az, ax):
	"""
	Calculate second order derivative for p(z,x,t)
	method: finite-difference
	"""
	# extend model to tmin=0 tmax=0 -- insert zeros in dimension t
	nz, nx = len(az), len(ax)
	dt = at[1] - at[0]
	# Modify p array in-place without copying
	p[:, 0] = 0.0
	p[:, -1] = 0.0
	_dt2 = 1./dt**2
	# Calculate second order derivative using vectorized operations
	p_dt_dt = (p[:, :-2] - 2 * p[:, 1:-1] + p[:, 2:]) * _dt2
	# Insert zeros at boundaries using array views
	p_dt_dt = np.concatenate([np.zeros((nz*nx, 1)), p_dt_dt, np.zeros((nz*nx, 1))], axis=1)

	return p_dt_dt

def second_order_derivative(p, at, az, ax):
	"""
	Calculate second order derivative for p(z,x,t)
	method: finite-difference
	"""
	# extend model to tmin=0 tmax=0 -- insert zeros in dimension t
	nz, nx = len(az), len(ax)
	dt = at[1] - at[0]
	# Modify p array in-place without copying
	p = cp.array(p)
	p[:, 0] = 0.0
	p[:, -1] = 0.0
	_dt2 = 1./dt**2
	# Calculate second order derivative using vectorized operations
	p_dt_dt = (p[:, :-2] - 2 * p[:, 1:-1] + p[:, 2:]) * _dt2
	# Insert zeros at boundaries using array views
	p_dt_dt = cp.concatenate([cp.zeros((nz*nx, 1)), p_dt_dt, cp.zeros((nz*nx, 1))], axis=1)

	return cp.asnumpy(p_dt_dt)

def adjoint_gradient(vel,d_obs,wsrc,zxsrc,zxrec,at,az,ax,next,device):
	"""
	Calculate gradiant using adjoint state method
	"""
	nsrc = zxsrc.shape[1]
	grads = np.zeros_like(vel)
	for i in range(nsrc):
		# print('Source', i, end='...   ')
		zxsrci = np.expand_dims(zxsrc[:,i],-1)
		p_fwd, d = prop2d(wsrc,zxsrci,zxrec,vel,at,az,ax,next,device)
		residual = d - d_obs[i] # residual shape (n_rec, nt) - residual.shape=(5,801)
		# Calculate back propagated p_back(z,x,t)
		p_back, _ = prop2d(np.flip(residual,axis=1), zxrec, zxsrci, vel, at, az, ax, next, device)
		p_back = np.flip(p_back,axis=1)
		# Calculate second order time derivative of p_fwd(z,x,t)
		p_dt_dt = second_order_derivative_cpu(p_fwd, at, az, ax)
		# Calculate gradiant
		grad = 2/vel**3  * np.sum(p_back * p_dt_dt, axis=1) # G.shape same as vel
		grads += grad

	return grads.flatten()

def misfit(vel,d_obs,wsrc,zxsrc,zxrec,at,az,ax,next,device):
	nsrc = zxsrc.shape[1]
	loss = 0
	for i in range(nsrc):
			zxsrci = np.expand_dims(zxsrc[:,i],-1)
			nz, nx = len(az), len(ax)
			if vel.ndim == 1: vel = np.reshape(vel.flatten(),(nz,nx))
			_, d = prop2d(wsrc,zxsrci,zxrec,vel,at,az,ax,next,device)
			residual = d - d_obs[i]
			J = 0.5 * np.sum(residual ** 2) # Residual is the half of the L2 norm square
			loss += J

	return loss