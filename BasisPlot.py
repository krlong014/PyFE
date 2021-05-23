import matplotlib.tri as pytri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import sqrt

fig = plt.figure()


tri = pytri.Triangulation([1, -sqrt(3)/2, -sqrt(3)/2], [0, 0.5, -0.5])
ref = pytri.UniformTriRefiner(tri)

phi0 = [1, 0, 0]
phi1 = [0, 1, 0]
phi2 = [0, 0, 1]
shadow = [0, 0, 0]

rtri, rphi0 = ref.refine_field(phi0, subdiv=4)
rtri, rphi1 = ref.refine_field(phi1, subdiv=4)
rtri, rphi2 = ref.refine_field(phi2, subdiv=4)

ax = fig.add_subplot(131, projection='3d')
ax.plot_trisurf(rtri, rphi0, cmap=cm.coolwarm)
ax.plot_trisurf(tri, shadow)
ax = fig.add_subplot(132, projection='3d')
ax.plot_trisurf(rtri, rphi1, cmap=cm.coolwarm)
ax.plot_trisurf(tri, shadow)
ax = fig.add_subplot(133, projection='3d')
ax.plot_trisurf(rtri, rphi2, cmap=cm.coolwarm)
ax.plot_trisurf(tri, shadow)
plt.show()
