import gym
import torch.nn as nn
import torch
import numpy as np
import random
import cv2
import pickle

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

def crop(frame:np.ndarray):
    '''No Upper Text-Noise Information'''
    image = frame[40:,:,:]
    return image

def resize(frame:np.ndarray, height:int=84, width:int=84):
    image = cv2.resize(frame, (height, width), interpolation=cv2.INTER_AREA)
    return image

class CNN(nn.Module):
    def __init__(self, in_channels:int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x)

class DuelingNetwork(nn.Module):
    def __init__(self, observation_space:gym.Space, n_frame_stack=3, out_actions=3):
        super().__init__()
        self.feature_extracter = CNN(n_frame_stack)

        frame = observation_space.sample()
        frame = crop(frame)
        frame = resize(frame)
        # stack_frame = np.stack([frame for _ in range(n_frame_stack)])# [H,W] -> [C,H,W]
        batch_stack_frame = np.expand_dims(np.transpose(frame, axes=(2,0,1)), axis=0) # [C,H,W] -> [B,C,H,W]

        with torch.no_grad():
            batch_stack_frame = torch.as_tensor(batch_stack_frame, dtype=torch.float32)
            n_flatten = self.feature_extracter(batch_stack_frame).shape[1]
        
        self.adv = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, out_actions),
        )
        
        self.value = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.feature_extracter(x)
        value = self.value(x)
        advantage = self.adv(x)
        mean_adv = torch.mean(advantage, dim=1, keepdim=True)
        q = value + advantage - mean_adv
        return q
    
class Agent():
    """Agent that acts randomly."""
    def __init__(self):
        env = env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)

        self.q_net = DuelingNetwork(env.observation_space, env.observation_space.shape[2], 3)
        self.q_net.load_state_dict(torch.load('112062546_hw2_data.py'))
        self.n_frame_skip = 16
        self.skip_count = 0


    def act(self, observation):
        #return 1
        if self.skip_count == 5927:
            self.skip_count = 0
        if self.skip_count % self.n_frame_skip == 0:
            obs = grayscale(observation)
            obs = resize(obs)
            obs = np.transpose(obs, axes=(2,0,1))
            obs = torch.tensor(obs, dtype=torch.float32, device=torch.device('cpu')).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(self.q_net(obs))
            self.action = action.item()

        self.skip_count += 1
        return self.action

if __name__ == '__main__':

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    done = False
    obs =env.reset()
    agent = Agent()
    total_reward = 0
    while not done:
        env.render()
        obs_, reward, done, info = env.step(agent.act(obs))
        total_reward+=reward        
        obs = obs_
    print(total_reward)


            
    