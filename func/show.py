"""
Functions to plot figures
"""
import matplotlib.pyplot as plt

def show_v(v):
  plt.figure(figsize=(6, 4))
  # plt.imshow(v, extent=[ax[0], ax[-1], az[-1], az[0]],vmin=2400,vmax=2600)
  plt.imshow(v)
  plt.colorbar()
  plt.scatter(zxsrc[1,:]*dx,zxsrc[0,:]*dz,marker='*',color='r',s=300,alpha=0.4)
  plt.scatter(zxrec[1,:]*dx,zxrec[0,:]*dz,marker='s',color='w',s=5,alpha=0.6)
  plt.xlabel('x position (m)', fontsize=labelsize)
  plt.ylabel('z position (m)', fontsize=labelsize)
  plt.title('Velocity field', fontsize=labelsize)
  plt.xticks(fontsize=labelsize-2)
  plt.yticks(fontsize=labelsize-2)
  plt.show()