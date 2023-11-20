#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:33:47 2021

@author: deniz
"""

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, DQN



def collect_data(env,size=20000):
    states = []
    actions = []
    rewards = []
    env.reset()
    for i in range(size):
        act=env.action_space.sample()
        obs,rew,done,_ =env.step(act)
        states.append(obs)
        actions.append(act)
        rewards.append(rew)
        if done:
            env.reset()
    return(pd.DataFrame(states))

    
   
env_source = gym.make('CartPole-v1')
env_target = gym.make('CartPole-v0')

source_data = collect_data(env_source,size=1000)
target_data = collect_data(env_target,size=1000)

source_data.to_csv('x1.csv')
target_data.to_csv('x2.csv')


plt.hist2d(source_data[0],source_data[1],bins=100,cmap='jet')
plt.title('MountainCarContinuous')
plt.xlabel('x')
plt.ylabel(r'$\dot{x}$')

plt.hist2d(target_data[2],target_data[3],bins=100,cmap='jet')
plt.title('CartPole0')
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot{x}$')

env = gym.make('MountainCarContinuous-v0')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000, log_interval=1)
model.save('ppo_mc_cont')

def train_model(env_name,max_step=100000):
    env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=max_step)
    model_name = 'PPO'+env_name
    model.save(model_name)
    
train_model('CartPole-v0')

obs=env.reset()
done=False
while not done:
    obs,rew,done,_=env.step(model.predict(obs)[0])
    env.render()
    
env.close()

from sklearn.metrics import pairwise_distances
import seaborn as sns
from sklearn.manifold import SpectralEmbedding
model = SpectralEmbedding(n_components=3, n_neighbors=45)
proj = model.fit_transform(source_data)

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
colors = {'news': 'green', 'religion': 'purple', 'fiction': 'red', 'government': 'black',
          'reviews': 'blue'}
ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
           c=data['label'].apply(lambda x: colors[x]))

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
           c=data['label'].apply(lambda x: colors[x]))
ax.view_init(4, -80)
plt.suptitle('Spectral Embedding of the Brown Corpus')
plt.show()


D = pairwise_distances(source_data)
plt.figure(figsize=(10,6))
sns.heatmap(D[:10, :10], cmap="coolwarm", annot=True);


from sklearn.manifold import MDS

model = MDS(n_components=4)
proj = model.fit_transform(source_data)
plt.scatter(proj[:, 0], proj[:, 1], cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
# plt.clim(-0.5, 9.5);