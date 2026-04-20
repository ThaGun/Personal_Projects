import gymnasium as gym
import pygame
import os
import torch
import itertools
import random
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from CP_policy import DQN
from CP_memory import ReplayMemory

RUN_DIR = r"D:/GitHub/ML projects & practice/Projects/ReinforcementLearning/DQN/DQN_Cartpole"
os.makedirs(RUN_DIR, exist_ok=True)

class Agent:
    def __init__(self, memory_size, mini_batch_size, network_sync_rate, epsilon_init, epsilon_decay, epsilon_min, learning_rate, discount_factor, stop_on_reward):
        # Memory parameters
        self.memory_size = memory_size
        self.mini_batch_size = mini_batch_size
        self.network_sync_rate = network_sync_rate

        # Epsilon greedy parameters
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Training parameters
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.stop_on_reward = stop_on_reward

    def run(self, is_training, render=False):
        # CartPole environment from gymnasium
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
        
        # State and action sizes
        '''
        State size: 4, Action size: 2 extracted for the policy network design
        '''
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n.item()

        policy_net = DQN(state_size, action_size)
        
        if is_training:
            # Memory
            memory = ReplayMemory(self.memory_size)
            # Epsilon greedy initilization
            epsilon = self.epsilon_init
            # Target network
            target_net = DQN(state_size, action_size)
            target_net.load_state_dict(policy_net.state_dict())
            # Step counter
            ''' Step counter to track on memory '''
            step = 0
            # Optimizer
            self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.learning_rate)
            # Best reward
            best_reward = -float("inf")

        else:
            policy_net.load_state_dict(torch.load(os.path.join(RUN_DIR, "best_model.pth")))
            policy_net.eval()
            memory= None
            epsilon = 0.0
        
        # History setup
        reward_per_episode = []
        epsilon_history = []

        # Time init
        start_time = datetime.now()
        
        '''
        Simulating the environment with infinite episodes for the training loop
        '''
        for episode in itertools.count(1):
            try:
                state, _ = env.reset()
                # Tensor conversion
                state = torch.tensor(state, dtype=torch.float32)
                terminated = False
                score = 0.0

                while (not terminated and score < self.stop_on_reward):
                    # Epsilon greedy inside loop
                    if is_training and random.random() < epsilon:
                        action = env.action_space.sample()
                        # Tensor conversion
                        action = torch.tensor(action, dtype=torch.int64)
                    else:
                        with torch.no_grad():
                            # 1 dim -> 2 dim with unsqueeze
                            # 2 dim -> 1 dim with squeeze
                            action = policy_net(state.unsqueeze(dim=0)).squeeze().argmax()
                    
                    # Environment update
                    '''
                    Updating the enveronment with the action from the policy network or sampled action from epsilon greedy
                    '''
                    next_state, reward, terminated, _, _ = env.step(action.item())

                    # Tensor conversion
                    next_state = torch.tensor(next_state, dtype=torch.float32)
                    reward = torch.tensor(reward, dtype=torch.float32)

                    # Memory update
                    if is_training:
                        memory.append((state, action, next_state, reward, terminated))
                        # step counter to track on memory
                        step += 1
                    
                    # State update
                    state = next_state
                    score += reward

                # print(f"Episode: {episode} Score: {score}")
                # Update reward history
                reward_per_episode.append(score)

                # Outside training loop 
                if is_training:
                    # Saving the best rewarded model for evaluation
                    if score > best_reward:
                        best_reward = score
                        print(f"Best reward: {best_reward}, Episode: {episode}")
                        torch.save(policy_net.state_dict(), (os.path.join(RUN_DIR, "best_model.pth")))
                
                # Graph to visualize and trach reward and epsilon decay
                current_time = datetime.now()
                if current_time - start_time > timedelta(seconds=5):
                    self.graph(reward_per_episode, epsilon_history)
                    start_time = current_time
                
                # Optimization with mini batch of saved memory
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_net, target_net)

                    # Epsilon greedy decay
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Update target network with policy network
                    if step > self.network_sync_rate:
                        target_net.load_state_dict(policy_net.state_dict())
                        step = 0

            except KeyboardInterrupt:
                print(f"Training interrupted by user with best score {score} in episode {episode}")
                env.close()
                break
            finally:
                env.close()
    
    def optimize(self, mini_batch, policy_net, target_net):
        
        # Extracting environment data from mini batch memory
        state, action, next_state, reward, terminated = zip(*mini_batch)
        state = torch.stack(state)
        action = torch.stack(action)
        next_state = torch.stack(next_state)
        reward = torch.stack(reward)
        terminated = torch.tensor(terminated, dtype=torch.float32)

        # Target q and current q
        with torch.no_grad():
            # Double DQN implementation
            best_action_from_policy = policy_net(next_state).argmax(dim=1)
            target_q = reward + (1 - terminated) * self.discount_factor * target_net(next_state).gather(dim=1, index=best_action_from_policy.unsqueeze(dim=1)).squeeze()
            '''
            feeding the target network with the next state and the best action from the policy network
            '''
        current_q = policy_net(state).gather(1, action.unsqueeze(dim=1)).squeeze()

        # Loss with MSE
        loss = self.loss_fn(current_q, target_q)

        # Optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def graph(self, reward_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(reward_per_episode))
        for x in range(len(reward_per_episode)):
            mean_rewards[x] = np.mean(reward_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)
        plt.plot(mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")

        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon Decay")
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(os.path.join(RUN_DIR, "graph.png"))
        plt.close(fig)


if __name__ == "__main__":
    agent = Agent(memory_size=100000,
                  mini_batch_size=32,
                  network_sync_rate=10,
                  epsilon_init=1.0,
                  epsilon_decay=0.995,
                  epsilon_min=0.0001,
                  learning_rate=0.01,
                  discount_factor=0.99,
                  stop_on_reward=100000)
    # Training phase
    # agent.run(is_training=True, render=False)

    # Evalute with best model
    agent.run(is_training=False, render=True)
    