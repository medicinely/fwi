"""
Functions for 2D wave propagation
"""
import numpy as np
import cupy as cp
from math import pi, sqrt, exp, sin, cos

precision = np.float32
precision_gpu = cp.float32

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

def prop2d_cpu(pwsrc,vel,at,az,ax,next):
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
	dz2   = 1./dz**2
	dx2   = 1./dx**2
	# Extend the model
	nze  = nz + 2*next2
	nxe  = nx + 2*next2
	vele = extend_model(vel,next2)
	# Shift the source by next
	p     = np.zeros([nze,nxe,nt], dtype=precision)
	pm    = np.zeros([nze,nxe], dtype=precision) # Previous wave field
	for it in range(1,nt-1): # From 1 to nt-1
		# sys.stdout.write(".")
		# sys.stdout.flush()
		# Second-order derivatives in z and x
		# + source term
		pp   = p[:,:,it]
		fact = (dt*vele[1+nabs:-1-nabs,1+nabs:-1-nabs])**2
		lapx = (pp[1+nabs:-1-nabs,0+nabs:-2-nabs] - \
				2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				pp[1+nabs:-1-nabs,2+nabs:-nabs])*dz2
		lapz = (pp[0+nabs:-2-nabs,1+nabs:-1-nabs] - \
				2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				pp[2+nabs:-nabs,1+nabs:-1-nabs])*dx2
		asrc = np.zeros([nze-2*nabs-2,nxe-2*nabs-2])
		asrc[next:-next,next:-next] = pwsrc[1:-1,1:-1,it]
		p[1+nabs:-1-nabs,1+nabs:-1-nabs,it+1] = \
			2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] - \
			pm[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
			(lapz + lapx + asrc)*fact
		pm = pp

		# One-way equation (bottom part)
		p[nze-1-nabs:nze,:nxe,it+1] = pp[nze-1-nabs:nze,:nxe,it] - \
			vele[nze-1-nabs:nze,:nxe]*dt/dz* \
			(p[nze-1-nabs:nze,:nxe,it]-p[nze-2-nabs:nze-1,:nxe,it])
		# One-way equation (top part)
		p[:1+nabs,:nxe,it+1] = p[:1+nabs,:nxe,it] + \
			vele[:1+nabs,:nxe]*dt/dz* \
			(p[1:2+nabs,:nxe,it]-p[:1+nabs,:nxe,it])
		# One-way equation (right part)
		p[:nze,nxe-1-nabs:nxe,it+1] = p[:nze,nxe-1-nabs:nxe,it] - \
			vele[:nze,nxe-1-nabs:nxe]*dt/dx* \
		(p[:nze,nxe-1-nabs:nxe,it] - p[:nze,nxe-2-nabs:nxe-1,it])
		# One-way equation (left part)
		p[:nze,:1+nabs,it+1] = p[:nze,:1+nabs,it] + \
			vele[:nze,:1+nabs]*dt/dx* \
			(p[:nze,1:2+nabs,it]-p[:nze,:1+nabs,it])
	# print(".")
#     print("min/max amplitudes of the wave field:",np.max(p),np.min(p))
	return p[next2:nze-next2,next2:nxe-next2,:]

def prop2d_gpu(pwsrc,vel,at,az,ax,next):
	"""
	2d wave propagation
	Resolution with finite differences
	Orders 2 in time and space
	with absorbing boundaries (Clayton and Engquist)
	Vectorial implementation (much faster)
	"""
	pwsrc,at,az,ax= cp.asarray(pwsrc),\
							  cp.asarray(at),cp.asarray(az),\
							  cp.asarray(ax)

	nabs  = 10
	next2 = nabs + next
	nt    = len(at)
	nz    = len(az)
	nx    = len(ax)
	dz    = az[1] - az[0]
	dx    = ax[1] - ax[0]
	dt    = at[1] - at[0]
	dz2   = 1./dz**2
	dx2   = 1./dx**2
	# Extend the model
	nze  = nz + 2*next2
	nxe  = nx + 2*next2
	vele = extend_model(vel,next2)
	vele = cp.asarray(vele)
	# Shift the source by next
	p     = cp.zeros([nze,nxe,nt], dtype=precision_gpu)
	pm    = cp.zeros([nze,nxe], dtype=precision_gpu) # Previous wave field
	for it in range(1,nt-1): # From 1 to nt-1
		# Second-order derivatives in z and x
		# + source term
		pp   = p[:,:,it]
		fact = (dt*vele[1+nabs:-1-nabs,1+nabs:-1-nabs])**2
		lapx = (pp[1+nabs:-1-nabs,0+nabs:-2-nabs] - \
				2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				pp[1+nabs:-1-nabs,2+nabs:-nabs])*dz2
		lapz = (pp[0+nabs:-2-nabs,1+nabs:-1-nabs] - \
				2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				pp[2+nabs:-nabs,1+nabs:-1-nabs])*dx2
		asrc = cp.zeros([nze-2*nabs-2,nxe-2*nabs-2])
		asrc[next:-next,next:-next] = pwsrc[1:-1,1:-1,it]
		p[1+nabs:-1-nabs,1+nabs:-1-nabs,it+1] = \
			2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] - \
			pm[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
			(lapz + lapx + asrc)*fact
		pm = pp

		# One-way equation (bottom part)
		p[nze-1-nabs:nze,:nxe,it+1] = p[nze-1-nabs:nze,:nxe,it] - \
			vele[nze-1-nabs:nze,:nxe]*dt/dz* \
			(p[nze-1-nabs:nze,:nxe,it]-p[nze-2-nabs:nze-1,:nxe,it])
		# One-way equation (top part)
		p[:1+nabs,:nxe,it+1] = p[:1+nabs,:nxe,it] + \
			vele[:1+nabs,:nxe]*dt/dz* \
			(p[1:2+nabs,:nxe,it]-p[:1+nabs,:nxe,it])
		# One-way equation (right part)
		p[:nze,nxe-1-nabs:nxe,it+1] = p[:nze,nxe-1-nabs:nxe,it] - \
			vele[:nze,nxe-1-nabs:nxe]*dt/dx* \
		(p[:nze,nxe-1-nabs:nxe,it] - p[:nze,nxe-2-nabs:nxe-1,it])
		# One-way equation (left part)
		p[:nze,:1+nabs,it+1] = p[:nze,:1+nabs,it] + \
			vele[:nze,:1+nabs]*dt/dx* \
			(p[:nze,1:2+nabs,it]-p[:nze,:1+nabs,it])
#     print("min/max amplitudes of the wave field:",np.max(p),np.min(p))
	return cp.asnumpy(p[next2:nze-next2,next2:nxe-next2,:])

def prop2d(wsrc, zxsrc, vel, at, az, ax, next, device):
	"""
	2d wave propagation with multiple sources
	"""
	if wsrc.ndim == 1: wsrc=np.array([wsrc]) # convert source wavelet from 1d to 2d
	zxsrc = np.array(zxsrc)
	nz, nx, nt = len(az),len(ax),len(at)
	pwsrc = np.zeros([nz, nx, nt]) # creat a initial p with zeros
	pwsrc[zxsrc[0,:], zxsrc[1,:], :wsrc.shape[1]] = wsrc # insert source wavelet
	if device=='gpu': p = prop2d_gpu(pwsrc, vel, at, az, ax, next) # propagate
	if device=='cpu': p = prop2d_cpu(pwsrc, vel, at, az, ax, next) # propagate

	return p

def new_prop2d_gpu(pwsrc,vel,at,az,ax,next):
	"""
	2d wave propagation
	Resolution with finite differences
	Orders 2 in time and space
	with absorbing boundaries (Clayton and Engquist)
	Vectorial implementation (much faster)
	"""
	pwsrc,at,az,ax= cp.asarray(pwsrc),\
							  cp.asarray(at),cp.asarray(az),\
							  cp.asarray(ax)
	nabs  = 10
	next2 = nabs + next
	nt    = len(at)
	nz    = len(az)
	nx    = len(ax)
	dz    = az[1] - az[0]
	dx    = ax[1] - ax[0]
	dt    = at[1] - at[0]   
	dz2   = 1./dz**2
	dx2   = 1./dx**2
	# Extend the model
	nze  = nz + 2*next2
	nxe  = nx + 2*next2
	vele = extend_model(vel,next2)
	vele = cp.asarray(vele)
	# Shift the source by next
	p_all = cp.zeros([nze,nxe,2]) # All states wave field
	pm = cp.zeros([nze,nxe]) # Previous state wave field
	pp = cp.zeros([nze,nxe])# Current state wave field
	for it in range(1,nt-1): # From 1 to nt-1
		fact = (dt*vele[1+nabs:-1-nabs,1+nabs:-1-nabs])**2
		lapx = (pp[1+nabs:-1-nabs,0+nabs:-2-nabs] - \
				2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				pp[1+nabs:-1-nabs,2+nabs:-nabs])*dz2
		lapz = (pp[0+nabs:-2-nabs,1+nabs:-1-nabs] - \
				2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				pp[2+nabs:-nabs,1+nabs:-1-nabs])*dx2
		asrc = cp.zeros([nze-2*nabs-2,nxe-2*nabs-2],dtype="float32")
		asrc[next:nze-2*nabs-2-next,next:nxe-2*nabs-2-next] = pwsrc[1:-1,1:-1,it]
		# One-way equation (center part)
		p0 = 2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] - \
				pm[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				(lapz + lapx + asrc)*fact 
		# One-way equation (bottom part)
		p1 = (pp[nze-1-nabs:nze,:nxe] - \
					vele[nze-1-nabs:nze,:nxe]*dt/dz* \
					(pp[nze-1-nabs:nze,:nxe]-\
					pp[nze-2-nabs:nze-1,:nxe]))[:,1+nabs:nxe-1-nabs]
		# One-way equation (top part)
		p2 = (pp[:1+nabs,:nxe] + \
					vele[:1+nabs,:nxe]*dt/dz* \
					(pp[1:2+nabs,:nxe]-pp[:1+nabs,:nxe]))[:,1+nabs:nxe-1-nabs]
		p012 = cp.vstack((p2,p0,p1))	# top-center-bottom
		# p012 = tf.concat([p2,p0,p1],axis=0) # top-center-bottom

		# One-way equation (right part)
		p3 = pp[:nze,nxe-1-nabs:nxe] - \
						vele[:nze,nxe-1-nabs:nxe]*dt/dx* \
						(pp[:nze,nxe-1-nabs:nxe] - pp[:nze,nxe-2-nabs:nxe-1])
		# One-way equation (left part)
		p4 = pp[:nze,:1+nabs] + \
						vele[:nze,:1+nabs]*dt/dx* \
						(pp[:nze,1:2+nabs]-pp[:nze,:1+nabs])
		p_next = cp.hstack((p4,p012,p3)) # all parts
		# p_next = tf.concat([p4,p012,p3],axis=1) # all parts
		p_all = cp.append(p_all,cp.expand_dims(p_next,-1),axis=-1)
		# p_all = tf.concat([p_all,tf.expand_dims(p_next,-1)],-1)
		# Update next state and save current state
		pm = pp
		pp = p_next
	p =  p_all[next2:nze-next2,next2:nxe-next2,:]

	return cp.asnumpy(p)