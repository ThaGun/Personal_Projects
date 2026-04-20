import gymnasium as gym
import pygame

env = gym.make("CartPole-v1", render_mode="human")

episodes = 10

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    terminated = False
    score = 0

    while not terminated:
        env.render()
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, _ = env.step(action)
        score += reward

    print(f"Episode: {episode} Score: {score}")

env.close()