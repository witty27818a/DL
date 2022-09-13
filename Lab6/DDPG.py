'''DLP DDPG Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
# import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from warnings import filterwarnings
filterwarnings("ignore")

class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim) # default mean = 0
        self.std = std if std else np.ones(dim) * .1 # default std = 0.1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        ## TODO ##
        transitions = random.sample(self.buffer, batch_size) # 從replay buffer裡面抽樣batch size個transitions
        return (torch.tensor(x, dtype = torch.float, device = device) # 包裝成tensors
                for x in zip(*transitions))


class ActorNet(nn.Module): # 給定環境狀態s，輸出動作mu (平均值因為是連續)
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        ## TODO ##
        self.layer1 = nn.Linear(state_dim, hidden_dim[0])
        self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.layer3 = nn.Linear(hidden_dim[1], action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        ## TODO ##
        output = self.relu(self.layer1(x))
        output = self.relu(self.layer2(output))
        output = self.tanh(self.layer3(output))
        return output


class CriticNet(nn.Module): # 給定環境狀態s和動作a，輸入預測的Q值
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x, action):
        x = self.critic_head(torch.cat([x, action], dim=1))
        return self.critic(x)


class DDPG:
    def __init__(self, args):
        '''actor和critic各自都有一組behavior網路和target網路'''
        # behavior network
        self._actor_net = ActorNet().to(args.device)
        self._critic_net = CriticNet().to(args.device)
        # target network
        self._target_actor_net = ActorNet().to(args.device)
        self._target_critic_net = CriticNet().to(args.device)
        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net.load_state_dict(self._critic_net.state_dict())
        ## TODO ##
        self._actor_opt = optim.Adam(self._actor_net.parameters(), lr = args.lra) #1e-3 # 優化器
        self._critic_opt = optim.Adam(self._critic_net.parameters(), lr = args.lrc) #1e-3 # 優化器
        # action noise
        self._action_noise = GaussianNoise(dim=2)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size # 64
        self.tau = args.tau # 0.005 # target網路軟更新權重，留一小部分新的behavior網路的，其他用之前target網路的
        self.gamma = args.gamma # 0.99 # 計算Q-target時的折舊因子

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        ## TODO ##
        with torch.no_grad():
            if noise: # if exploration noises are applied
                actions = self._actor_net(torch.from_numpy(state).view(1, -1).to(self.device)) + \
                    torch.from_numpy(self._action_noise.sample()).view(1, -1).to(self.device) # 把狀態塞給actor網路，決定出動作，訓練時加上噪聲來增加變化
            else:
                actions = self._actor_net(torch.from_numpy(state).view(1, -1).to(self.device))
        return actions.cpu().numpy().squeeze()

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state,
                            [int(done)])

    def update(self):
        # update the behavior networks
        self._update_behavior_network(self.gamma)
        # update the target networks
        self._update_target_network(self._target_actor_net, self._actor_net,
                                    self.tau)
        self._update_target_network(self._target_critic_net, self._critic_net,
                                    self.tau)

    def _update_behavior_network(self, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, self._target_actor_net, self._target_critic_net
        actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        # critic loss
        ## TODO ##
        q_value = critic_net(state, action) # 用critic網路計算目前時刻的Q-value
        with torch.no_grad():
           a_next = target_actor_net(next_state) # 用actor的target網路判斷出下一狀態下，要執行什麼action
           q_next = target_critic_net(next_state, a_next) # 再用critic的target網路判斷下一狀態和這個action，的Q值是多少
           q_target = reward + gamma * q_next * (1 - done) # 用獎勵和折舊因子等，推算回本時刻應該有的Q-target值
        criterion = nn.MSELoss() # 用現在動作的q-value，和q-target之間的MSE當作loss
        critic_loss = criterion(q_value, q_target)
        # optimize critic
        actor_net.zero_grad()
        critic_net.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ## update actor ##
        # actor loss
        ## TODO ##
        action = actor_net(state) # 給定狀態，用actor網路決定行動action
        actor_loss = -critic_net(state, action).mean() # 用critic網路給出的期望梯度方向來更新，加上負號因為是loss要minimize
        # optimize actor
        actor_net.zero_grad()
        critic_net.zero_grad()
        actor_loss.backward()
        actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            ## TODO ##
            target.data.copy_((1 - tau) * target.data + tau * behavior.data) # target網路軟更新權重，留一小部分新的behavior網路的，其他用之前target網路的

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic': self._target_critic_net.state_dict(),
                    'actor_opt': self._actor_opt.state_dict(),
                    'critic_opt': self._critic_opt.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net.load_state_dict(model['critic'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net.load_state_dict(model['target_critic'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt.load_state_dict(model['critic_opt'])


def train(args, env, agent, writer):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup: #10000iter
                action = env.action_space.sample() #在此之前都隨機抽action，讓模型有explore的機會
            else:
                action = agent.select_action(state) #狀態加上噪聲後，給actor網路選動作
            # execute action
            next_state, reward, done, _ = env.step(action) # 執行某個action，回傳新狀態，獎勵和是否Game over
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup: # 10000iter之前都讓模型亂玩沒差，
                agent.update() # 之後的行動再用來真正更新參數

            state = next_state # 更新遊戲盤面(環境)狀態
            total_reward += reward # 新增上個動作的獎勵
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()# 刷新遊戲，重新開始
        ## TODO ##
        for t in itertools.count(start = 1): # 一圈玩一次
            if args.render:
                env.render() # 渲染出此刻的遊戲畫面給你看
            
            action = agent.select_action(state, noise = False) # 測試時動作action直接用平均值，不需要噪聲添亂
            next_state, reward, done, _ = env.step(action) # 執行某個action，回傳新狀態，獎勵和是否Game over

            state = next_state # 更新遊戲盤面(環境)狀態
            total_reward += reward # 新增上個動作的獎勵

            if done: # Game over
                # writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print(f"Total Reward {n_episode + 1}: {total_reward:.2f}") # 本次總得分
                rewards.append(total_reward)
                break # 遊戲結束，跳出迴圈下一回合
    
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='D:/DL_lab6_params/ddpg_nozip.pth')
    parser.add_argument('--logdir', default='D:/DL_lab6_params/log/ddpg')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=2000, type=int) # 2000, originally 1200
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=500000, type=int)
    parser.add_argument('--lra', default=1e-3, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.005, type=float)
    # test
    parser.add_argument('--test_only', default = True) # action='store_true'
    parser.add_argument('--render', default = True) # action = 'store_true'
    parser.add_argument('--seed', default=20200519, type=int) #20200504
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLanderContinuous-v2')
    agent = DDPG(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
