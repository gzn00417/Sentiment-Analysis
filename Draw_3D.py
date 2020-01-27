import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

f=open("record10.txt","r")
data=[]
for line in f:
    nums=line.strip().split()
    part=[]
    for x in nums:
        part.append(float(x))
    data.append(part)

x=[]
y=[]
z=[]
for i in range(10):
    for j in range(10):
        x.append(float(i)/10)
        y.append(float(j)/10)
        z.append(data[i][j])

fig=plt.figure()
#ax=Axes3D(fig)
#ax.scatter(x,y,z)
ax=fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x,y,z,linewidth=0.1, antialiased=True,cmap='rainbow')

ax.set_xlabel('X', fontdict={'size': 17, 'color': 'orange'})
ax.set_ylabel('Y', fontdict={'size': 17, 'color': 'orange'})
ax.set_zlabel('Z', fontdict={'size': 17, 'color': 'purple'})

plt.show()