import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym
import turtle

print(gym.__file__)

v = torch.arange(9)
v = v.view(3, 3)

# print(v)

t = torch.ones(2, 1, 2, 1)  # Size 2x1x2x1
r = torch.squeeze(t)     # Size 2x2
# r = torch.squeeze(t, 1)  # Squeeze dimension 1: Size 2x2x1

# print(t)
# print(r)

# Un-squeeze a dimension
x = torch.Tensor([1, 2, 3])
r = torch.unsqueeze(x, 0)       # Size: 1x3
r = torch.unsqueeze(x, 1)       # Size: 3x1

# print(x, t, r)

env = gym.make("Taxi-v2")
env.reset()

# print('SS:', env.observation_space.n)

# env.render()

# print(env.action_space.n)

# env.env.s = 114
# env.render()

# env.step(1)
# env.render()
Q = np.zeros([env.observation_space.n, env.action_space.n])
print('size: ', env.observation_space.n)

G = 0
alpha = 0.5

for episode in range(1, 1001):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
        action = np.argmax(Q[state])  # 1
        state2, reward, done, info = env.step(action)  # 2
        Q[state, action] += alpha * \
            (reward + np.max(Q[state2]) - Q[state, action])  # 3
        G += reward
        state = state2
    # if episode % 50 == 0:
        #print('Episode {} Total Reward: {}'.format(episode, G))
        # env.render()

turtle.shape("turtle")
turtle.forward(25)
turtle.exitonclick()
# state = env.reset()
# counter = 0
# reward = None
# while reward != 20:
#     state, reward, done, info = env.step(env.action_space.sample())
#     counter += 1

# print(counter)
