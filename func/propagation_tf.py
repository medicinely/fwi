"""
Tensorflow functions for 2D wave propagation
"""
import numpy as np
import tensorflow as tf

def prop2d_tf(vel,pwsrc,at,az,ax,next):
	nabs  = tf.constant(10)
	next2 = nabs + next
	nt    = tf.size(at)
	nz    = tf.size(az)
	nx    = tf.size(ax)
	az = tf.convert_to_tensor(az,dtype="float32")
	ax = tf.convert_to_tensor(ax,dtype="float32")
	at = tf.convert_to_tensor(at,dtype="float32")
	dz    = az[1] - az[0]
	dx    = ax[1] - ax[0]
	dt    = at[1] - at[0]   
	dz2   = 1./dz**2
	dx2   = 1./dx**2
	# Extend the model
	nze  = nz + 2*next2
	nxe  = nx + 2*next2
	########################################################
	#				    EXTEND MODEL				       #
	########################################################
	nze  = nz + 2*next2
	nxe  = nx + 2*next2
	vele = tf.zeros([nze,nxe],dtype="float32")
 
	# Central part
	vele = replace(vele,vel,[next2,nze-next2],[next2,nxe-next2])
 
	# Top and bottom
	vele = replace(vele,tf.repeat([vel[0,:]],next2,axis=0),
								 [0,next2],[next2,nxe-next2])
	vele = replace(vele,tf.repeat([vel[-1,:]],next2,axis=0),
	               [nze-next2,nze],[next2,nxe-next2]) 
	# Left and right
	vele = replace(vele,tf.transpose(tf.repeat([vel[:,0]],next2,axis=0)),
	                  [next2,nze-next2],[0,next2])
	vele = replace(vele,tf.transpose(tf.repeat([vel[:,-1]],next2,axis=0)),
	               		[next2,nze-next2],[nxe-next2,nxe])
	# Corners
	vele = replace(vele,tf.fill([next2, next2], vel[0,0]),
	               [0,next2],[0,next2])
	vele = replace(vele,tf.fill([next2, next2], vel[0,-1]),
	               [0,next2],[nxe-next2,nxe])
	vele = replace(vele,tf.fill([next2, next2], vel[-1,0]),
	               [nze-next2,nze],[0,next2])
	vele = replace(vele,tf.fill([next2, next2], vel[-1,-1]),
	               [nze-next2,nze],[nxe-next2,nxe])
	# vele = extend_model(vel,next2)
  ########################################################
	#
	########################################################
	# Shift the source by next
	p_all = tf.zeros([nze,nxe,2]) # All states wave field
	pm = tf.zeros([nze,nxe]) # Previous state wave field
	pp = tf.zeros([nze,nxe])# Current state wave field
	for it in tf.range(1,nt-1): # From 1 to nt-1
		fact = (dt*vele[1+nabs:-1-nabs,1+nabs:-1-nabs])**2
		lapx = (pp[1+nabs:-1-nabs,0+nabs:-2-nabs] - \
				2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				pp[1+nabs:-1-nabs,2+nabs:-nabs])*dz2
		lapz = (pp[0+nabs:-2-nabs,1+nabs:-1-nabs] - \
				2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				pp[2+nabs:-nabs,1+nabs:-1-nabs])*dx2
		asrc = tf.zeros([nze-2*nabs-2,nxe-2*nabs-2],dtype="float32")
		asrc = replace(asrc,pwsrc[1:-1,1:-1,it],
		              [next,nze-2*nabs-2-next],
               		[next,nxe-2*nabs-2-next])
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
		p012 = tf.concat([p2,p0,p1],axis=0) # top-center-bottom
		# One-way equation (right part)
		p3 = pp[:nze,nxe-1-nabs:nxe] - \
						vele[:nze,nxe-1-nabs:nxe]*dt/dx* \
						(pp[:nze,nxe-1-nabs:nxe] - pp[:nze,nxe-2-nabs:nxe-1])
		# One-way equation (left part)
		p4 = pp[:nze,:1+nabs] + \
						vele[:nze,:1+nabs]*dt/dx* \
						(pp[:nze,1:2+nabs]-pp[:nze,:1+nabs])
		p_next = tf.concat([p4,p012,p3],axis=1) # all parts
		p_all = tf.concat([p_all,tf.expand_dims(p_next,-1)],-1)
		# Update next state and save current state
		pm = pp
		pp = p_next
	return p_all[next2:nze-next2,next2:nxe-next2,:]



def simulate_obs(vel, wsrc, zxsrc, at, az, ax, next, zxrec):
	"""
	2d wave propagation with multiple sources
	"""
	nz,nx,nt = len(az),len(ax),len(at)
	vel = tf.convert_to_tensor(vel,dtype="float32")
	next = tf.constant(next)
	if wsrc.ndim == 1: wsrc=np.array([wsrc]) # convert source wavelet from 1d to 2d
	pwsrc = np.zeros([nz, nx, nt]) # creat a initial p with zeros
	pwsrc[zxsrc[0,:], zxsrc[1,:], :wsrc.shape[1]] = wsrc	# insert source wavelet
	pwsrc = tf.convert_to_tensor(pwsrc,dtype="float32")
	# p = prop2d_tf(vel,pwsrc,at,az,ax,next) # propagate

	############################ Prop2D ##########################
	nabs  = tf.constant(10)
	next2 = nabs + next
	nt    = tf.size(at)
	nz    = tf.size(az)
	nx    = tf.size(ax)
	az = tf.convert_to_tensor(az,dtype="float32")
	ax = tf.convert_to_tensor(ax,dtype="float32")
	at = tf.convert_to_tensor(at,dtype="float32")
	dz    = az[1] - az[0]
	dx    = ax[1] - ax[0]
	dt    = at[1] - at[0]   
	dz2   = 1./dz**2
	dx2   = 1./dx**2
	# Extend the model
	nze  = nz + 2*next2
	nxe  = nx + 2*next2
	#########################  EXTEND MODEL ########################
	nze  = nz + 2*next2
	nxe  = nx + 2*next2
	vele = tf.zeros([nze,nxe],dtype="float32")

	velnp = tf.convert_to_tensor(vel.numpy()) # creat velnp to stop recording
	# Top and bottom
	vele = replace(vele,tf.repeat([velnp[0,:]],next2,axis=0),
									[0,next2],[next2,nxe-next2])
	vele = replace(vele,tf.repeat([velnp[-1,:]],next2,axis=0),
					[nze-next2,nze],[next2,nxe-next2]) 
	# Left and right
	vele = replace(vele,tf.transpose(tf.repeat([velnp[:,0]],next2,axis=0)),
						[next2,nze-next2],[0,next2])
	vele = replace(vele,tf.transpose(tf.repeat([velnp[:,-1]],next2,axis=0)),
						[next2,nze-next2],[nxe-next2,nxe])
	# Corners
	vele = replace(vele,tf.fill([next2, next2], velnp[0,0]),
					[0,next2],[0,next2])
	vele = replace(vele,tf.fill([next2, next2], velnp[0,-1]),
					[0,next2],[nxe-next2,nxe])
	vele = replace(vele,tf.fill([next2, next2], velnp[-1,0]),
					[nze-next2,nze],[0,next2])
	vele = replace(vele,tf.fill([next2, next2], velnp[-1,-1]),
					[nze-next2,nze],[nxe-next2,nxe])
	# Central part
	vele = replace(vele,vel,[next2,nze-next2],[next2,nxe-next2])
  	###############################################################
	# Shift the source by next
	p_all = tf.zeros([nze,nxe,2]) # All states wave field
	pm = tf.zeros([nze,nxe]) # Previous state wave field
	pp = tf.zeros([nze,nxe])# Current state wave field
	for it in tf.range(1,nt-1): # From 1 to nt-1
		fact = (dt*vele[1+nabs:-1-nabs,1+nabs:-1-nabs])**2
		lapx = (pp[1+nabs:-1-nabs,0+nabs:-2-nabs] - \
				2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				pp[1+nabs:-1-nabs,2+nabs:-nabs])*dz2
		lapz = (pp[0+nabs:-2-nabs,1+nabs:-1-nabs] - \
				2.*pp[1+nabs:-1-nabs,1+nabs:-1-nabs] + \
				pp[2+nabs:-nabs,1+nabs:-1-nabs])*dx2
		asrc = tf.zeros([nze-2*nabs-2,nxe-2*nabs-2],dtype="float32")
		asrc = replace(asrc,pwsrc[1:-1,1:-1,it],
		              [next,nze-2*nabs-2-next],
               		[next,nxe-2*nabs-2-next])
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
		p012 = tf.concat([p2,p0,p1],axis=0) # top-center-bottom
		# One-way equation (right part)
		p3 = pp[:nze,nxe-1-nabs:nxe] - \
						vele[:nze,nxe-1-nabs:nxe]*dt/dx* \
						(pp[:nze,nxe-1-nabs:nxe] - pp[:nze,nxe-2-nabs:nxe-1])
		# One-way equation (left part)
		p4 = pp[:nze,:1+nabs] + \
						vele[:nze,:1+nabs]*dt/dx* \
						(pp[:nze,1:2+nabs]-pp[:nze,:1+nabs])
		p_next = tf.concat([p4,p012,p3],axis=1) # all parts
		p_all = tf.concat([p_all,tf.expand_dims(p_next,-1)],-1)
		# Update next state and save current state
		pm = pp
		pp = p_next
	p =  p_all[next2:nze-next2,next2:nxe-next2,:]

	########### Calculate observation at receivers position ##############
	nrec = zxrec.shape[1]
	d = tf.Variable(tf.zeros([nrec,nt]))
	for n in range(nrec):
		d = replace_1d(d,p[zxrec[0,n], zxrec[1,n], :],n,[0,nt])
	return d

# Code to replace tensor elements in tensorflow

def idx_to_replace(z1,z2,x1,x2):
  arr = []
  for i in range(z1,z2):
    for j in range(x1,x2):
      arr.append([i,j])
  return arr

def replace(tensor,arr,iz,ix):
	"""
	tensor: tensor to replace the elements
	arr: the array to insert
	iz,ix: the index of z and x
	"""
	replaced = tf.tensor_scatter_nd_update(
		tensor, 
    idx_to_replace(iz[0],iz[1],ix[0],ix[1]),
    tf.reshape(arr, [-1]))
	return replaced

def idx_to_replace_1d(n,t1,t2):
  arr = []
  for t in range(t1,t2):
      arr.append([n,t])
  return arr

def replace_1d(tensor,arr,n,it):
	"""
	tensor: tensor to replace the elements
	arr: the array to insert
	iz,ix: the index of z and x
	"""
	replaced = tf.tensor_scatter_nd_update(
		tensor, 
    idx_to_replace_1d(n,it[0],it[1]),
    tf.reshape(arr, [-1]))
	return replaced