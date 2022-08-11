import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import ngsolve as ngs

from opencmp.diffuse_interface import interface, mesh_helpers
from opencmp.helpers.ngsolve_ import ngsolve_to_numpy

N = [80, 80, 80]
scale = [5.0, 5.0, 2.6]
offset = [2.5, 2.5, 0.2]
dim = 3
interp_ord = 2

mesh = mesh_helpers.get_Netgen_nonconformal(N, scale, offset, dim, True)
mesh = ngs.Mesh(mesh)

fes = ngs.H1(mesh, order=interp_ord)

stl_filename = 'dim_dir/led.stl'
boundary_lst, bounds_lst = mesh_helpers.get_stl_faces(stl_filename)
binary = interface.get_binary_3d(boundary_lst, N, scale, offset, mnum=1, close=False)

data_filename = 'output/poisson_sol/poisson_0.0.sol'
data_gfu = ngs.GridFunction(fes)
data_gfu.Load(data_filename)

x = np.linspace(-2.5, 2.5, N[0])
y = np.linspace(-2.5, 2.5, N[1])
xx, yy = np.meshgrid(x, y)

data = ngsolve_to_numpy(mesh, data_gfu, N, scale, offset, dim, None)[0] * binary

z_grid = [20, 40, 60, 70]
z_vals = [z*(scale[2]/N[2]) - offset[2] for z in z_grid]
zz = [data[:,:,z] for z in z_grid]

mymap = cm.get_cmap('hot_r')

fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
for i in range(len(z_grid)):
    Z = np.zeros(xx.shape) + z_vals[i]
    t = zz[i]
    tt = mymap(t)
    surf = ax.plot_surface(xx, yy, Z, facecolors=tt, linewidth=0, alpha=0.5, rcount=N[0], ccount=N[1])

ax.invert_zaxis()
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
m = cm.ScalarMappable(cmap='hot_r')
m.set_array([298, 305])
cbaxes = fig.add_axes([0.85, 0.35, 0.03, 0.4]) 
cbar = plt.colorbar(m, cax = cbaxes)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Temperature (K)', rotation=270, labelpad=30, fontsize=20)
plt.show()
