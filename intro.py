#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:55:41 2021

@author: deniz
"""
# import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.core.pylabtools import figsize
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
figsize(12, 6)

# display multiple outputs within a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all";

# ignore warnings
import warnings
warnings.filterwarnings('ignore');



# plot unit circle in R^2
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
ax.add_patch(circ)
circ = plt.Circle((0, 0), radius=0.5, edgecolor='black', facecolor='None', linewidth=3, linestyle='--')
ax.add_patch(circ)

# sample within unit sphere in R^2
n = 1000
theta = np.random.uniform(0, 2*math.pi, n)
u = np.random.uniform(0, 1, n)
r = np.sqrt(u)
x = r * np.cos(theta)
y = r * np.sin(theta)
ax.scatter(x, y, s=10, alpha=1)

# proportion within neighborhood
p = np.sum((x**2 + y**2) <= 0.5**2) / n
ax.scatter(0, 0, s=20, c='black')
plt.title("% in neighborhood = " + str(p))

# plot unit sphere in R^3
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_aspect("auto")
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="black", alpha=0.5)
ax.plot_wireframe(0.5*x, 0.5*y, 0.5*z, color="black")

# uniform sampling
phi = np.random.uniform(0, 2*math.pi, n)
costheta = np.random.uniform(-1, 1, n)
u = np.random.uniform(0, 1, n)

theta = np.arccos(costheta)
r = u**(1/3)
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
ax.scatter(x, y, z, alpha=0.4, s=10)

# proportion within neighborhood
p = np.sum((x**2 + y**2 + z**2) <= 0.5**2) / n
ax.scatter([0], [0], [0], color="black", s=100)
plt.title("% in neighborhood = " + str(p));


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
mnist.data.shape



fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    jj = axi.imshow(mnist.data.iloc[1250 * i].values.reshape(28, 28), cmap='gray');
    
    
# explained variance from principal components
from sklearn.decomposition import PCA
# take subset of mnist data (1/10 of observations)
data = mnist.data[::5]
model = PCA()
proj = model.fit_transform(data)
cumsum = np.cumsum(model.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
plt.figure(figsize=(8,5))
plt.plot((0, 800), (0.95, 0.95), 'r--')
plt.plot(cumsum)
plt.xlim(0,200)
plt.style.use('seaborn')
plt.title('PCA Explained Variance')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
print('95% Explained Variance with ' + str(d) + ' Principal Components');


plt.subplot(2,2,1)
plt.imshow(np.mean(mnist.data).to_numpy().reshape(28,28), cmap='jet')


plt.subplot(2,2,2)
plt.imshow(model.mean_.reshape(28,28), cmap='jet')


plt.subplot(2,2,3)
plt.imshow((np.mean(mnist.data).to_numpy()-model.mean_).reshape(28,28), cmap='jet')

 
plt.subplot(2,2,4)
plt.imshow((model.mean_- np.mean(mnist.data).to_numpy()).reshape(28,28), cmap='jet')   
    
from stable_baselines3 import DDPG
