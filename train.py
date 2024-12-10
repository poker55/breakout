import torch
import torch.optim as optim
import time
from environment import BreakoutEnv
from policy import PolicyNetwork
from constants import *
import pygame
from torch.distributions import Normal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def train_agent():
    # Set up the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Games')
    ax.set_ylabel('Game Reward')
    ax.set_title('Learning Progress')
    
    # Initialize lists to store data
    games = []
    rewards = []
    smoothed_rewards = []
    line_raw, = ax.plot([], [], 'b-', alpha=0.3, label='Raw Rewards')
    line_smooth, = ax.plot([], [], 'b-', label='Smoothed Rewards')
    ax.legend()
    
    try:
        env = BreakoutEnv(render_mode="human")
        state_size = len(env.reset())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        policy = PolicyNetwork(state_size).to(device)
        #optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        optimizer = optim.SGD(policy.parameters(), lr=LEARNING_RATE)  # Changed from Adam to SGD
        running_reward = 0
        max_steps_per_game = 100000

        for game_num in range(MAX_EPISODES):
            game_log_probs = []
            game_rewards = []
            accumulated_game_reward = 0
            state = env.reset()
            steps_in_game = 0
            
            while steps_in_game < max_steps_per_game:
                steps_in_game += 1
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                mean_std = policy(state_tensor)
                mean, log_std = mean_std[:, 0], mean_std[:, 1]
                std = torch.exp(log_std)
                
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = torch.clamp(action, -1, 1)
                
                next_state, reward, done = env.step(action.item())
                
                ball_x = next_state[1] * SCREEN_WIDTH
                paddle_x = next_state[0] * SCREEN_WIDTH
                paddle_center = paddle_x + PADDLE_WIDTH/2
                positioning_reward = 1.0 / (1.0 + abs(ball_x - paddle_center) / SCREEN_WIDTH)
                reward += positioning_reward * 5
                
                game_log_probs.append(log_prob)
                game_rewards.append(reward)
                accumulated_game_reward += reward
                
                if done:
                    print(f"Game {game_num + 1}/{MAX_EPISODES} | Score: {env.score} | Steps: {steps_in_game} | Game Reward: {accumulated_game_reward:.1f} | Running Reward: {running_reward:.1f}")
                    
                    # Update plot data
                    games.append(game_num + 1)
                    rewards.append(accumulated_game_reward)
                    
                    # Calculate smoothed rewards
                    if len(rewards) > 0:
                        smoothed_rewards = np.convolve(rewards, np.ones(min(len(rewards), 50))/min(len(rewards), 50), mode='valid')
                    
                    # Update plot
                    line_raw.set_data(games, rewards)
                    if len(smoothed_rewards) > 0:
                        smooth_x = games[len(games)-len(smoothed_rewards):]
                        line_smooth.set_data(smooth_x, smoothed_rewards)
                    
                    # Adjust plot limits
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.01)
                    
                    break
                    
                state = next_state
            
            if len(game_rewards) > 0:
                try:
                    R = 0
                    returns = []
                    for r in game_rewards[::-1]:
                        R = r + GAMMA * R
                        returns.insert(0, R)
                    
                    returns = torch.FloatTensor(returns).to(device)
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                    
                    policy_loss = []
                    for log_prob, R in zip(game_log_probs, returns):
                        policy_loss.append(-log_prob * R)
                    
                    optimizer.zero_grad()
                    policy_loss = torch.stack(policy_loss).sum()
                    policy_loss.backward()
                    optimizer.step()
                    
                    running_reward = 0.05 * accumulated_game_reward + (1 - 0.95) * running_reward
                    
                except Exception as e:
                    print(f"Error during policy update: {e}")
                    continue
            
            if (game_num + 1) % 50 == 0:
                torch.save({
                    'game': game_num,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'running_reward': running_reward,
                }, f'breakout_pg_checkpoint_{game_num + 1}.pt')
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        torch.save({
            'game': game_num,
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'running_reward': running_reward,
        }, 'breakout_pg_interrupted.pt')
    finally:
        print("\nTraining completed")
        plt.ioff()
        plt.show()