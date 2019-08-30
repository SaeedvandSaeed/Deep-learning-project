# Test Github
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import Robot_Environment
from time import sleep

env = Robot_Environment.CartPoleEnv()
# gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *********************

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# **************************


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

# **************************


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_location(screen_width, screen_height):
    world_width = env.screen_width
    world_height = env.screen_height
    scale_w = screen_width / world_width
    scale_h = screen_height / world_height
    # Midle of object
    return (int(env.pos_x_object * scale_w), int(env.pos_y_object * scale_h))


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render1(mode='rgb_array')  # .transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    screen_height, screen_width, _ = screen.shape

    obj_location_x, obj_location_y = get_location(screen_width, screen_height)

    screen = screen[int((screen_height - obj_location_y - (env.objheight * 2))):
                    int((screen_height - obj_location_y + (env.objheight * 2))),
                    int(obj_location_x - (env.objwidth * 1.2)):
                    int(obj_location_x + (env.objwidth * 1.2))]

    # print(screen_width, screen_height,  obj_location_y)

    screen = screen.transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    # print(screen_width, screen_height,  obj_location_y)

    # The implementation are for sscaling the captured size of he piucture form
    # the drame that comes from environment, the size is become diffeerenetr that we have to
    # resize it to the desired one, themn we have to scale ior to the tnsor size

    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)

    # plt.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy())

    return resize(screen).unsqueeze(0).to(device)


env.reset1(randomness=False)


# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy())
# plt.title('Vision')
# plt.show()

# **************************

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 10
MODEL_SAVE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = 5  # @@@@@@@@@@

# actions_robot = {'PathPattern': 0, 'LeftHandAct': 0, 'RightHandAct': 0}

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

# **************************


def actoins_map(action_number):
    if(action_number == 0):
        return {'PathPattern': 0, 'LeftHandAct': 0, 'RightHandAct': 0}
    elif(action_number == 1):
        return {'PathPattern': 0, 'LeftHandAct': 0, 'RightHandAct': 1}
    elif(action_number == 2):
        return {'PathPattern': 0, 'LeftHandAct': 1, 'RightHandAct': 0}
    elif(action_number == 3):
        return {'PathPattern': 0, 'LeftHandAct': 2, 'RightHandAct': 0}
    elif(action_number == 4):
        return {'PathPattern': 0, 'LeftHandAct': 0, 'RightHandAct': 2}

# **************************


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # return actoins_map((4))
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            # random_action = random.randrange(n_actions)
            # return torch.tensor([[random_action]], device=device, dtype=torch.long), action_dic

            best_action = policy_net(state).max(1)[1].view(1, 1)
            print(best_action)
            action_number = best_action.numpy()[0][0]
            action_dic = actoins_map(action_number)
            return best_action, action_dic
    else:
        print('random')
        # return {'PathPattern': 0, 'LeftHandAct': 2, 'RightHandAct': 1}
        # return actoins_map(random.randrange(n_actions))
        random_action = random.randrange(n_actions)
        action_dic = actoins_map(random_action)
        return torch.tensor([[random_action]], device=device, dtype=torch.long), action_dic
        # return {'PathPattern': 0, 'LeftHandAct': random.randrange(3), 'RightHandAct': random.randrange(3)}


episode_results = []


def plot_durations():
    # return  # xxxxxxxxxxxx
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_results, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

# ****************************


def optimize_model():
    if len(memory) < BATCH_SIZE:
        # print(memory.sample(1))
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = \
        (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# *****************************


num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset1(randomness=False)
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        actions_robot, action_dic = select_action(state)
        state_object, reward1, done1, _ = env.step1(action_dic, False)

        #print('Rw', reward1)
        reward1 = torch.tensor([(reward1)], device=device)

        #print('action', actions_robot)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done1:
            next_state = current_screen - last_screen
        else:
            next_state = None
        # actions_robot
        action = torch.tensor(actions_robot, device=device, dtype=torch.long)

        # batch_action.long()
        # a.()

        # Store the transition in memory
        memory.push(state, action, next_state, reward1.float())

        # Move to the next state
        state = next_state
        # print(state)
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done1:
            episode_results.append(abs(env.ori_object / 100))  # (t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # if i_episode % MODEL_SAVE == 0:
    #     torch.save(policy_net.state_dict(), PATH)
    #     torch.save(target_net.state_dict(), PATH)

    print('episode:', i_episode)

print('Complete')
env.render1()
env.close()
plt.ioff()
plt.show()
