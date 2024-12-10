import torch
import torch.nn as nn
import numpy as np
from environment import BreakoutEnv
from policy import PolicyNetwork
from constants import *
import matplotlib.pyplot as plt
import torch.distributions as dist
import sys
import pygame

class SafeBreakoutEnv(BreakoutEnv):
    """Wrapper around BreakoutEnv with additional safety checks"""
    def __init__(self, render_mode=None):
        try:
            super().__init__(render_mode=render_mode)
        except Exception as e:
            print(f"Error initializing environment: {e}")
            sys.exit(1)

    def step(self, action):
        try:
            return super().step(action)
        except Exception as e:
            print(f"Error in environment step: {e}")
            return self.reset(), 0, True

    def reset(self):
        try:
            return super().reset()
        except Exception as e:
            print(f"Error in environment reset: {e}")
            return np.zeros(6, dtype=np.float32)  # Return safe default state

class SimplifiedMHOptimizer:
    def __init__(self, policy, sigma=0.005):  # Reduced sigma
        self.policy = policy
        self.sigma = sigma
        self.device = next(policy.parameters()).device

    def evaluate_policy(self, env, max_steps=500):  # Reduced max steps
        total_reward = 0
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            try:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.policy(state_tensor)
                    action = torch.tanh(output[:, 0])  # Simplified action selection
                
                next_state, reward, done = env.step(action.item())
                total_reward += reward
                state = next_state
                steps += 1
                
            except Exception as e:
                print(f"Error in policy evaluation: {e}")
                return -1000  # Return a bad score on error
                
        return total_reward

    def update_parameters(self, params):
        try:
            idx = 0
            for p in self.policy.parameters():
                numel = p.numel()
                new_data = params[idx:idx + numel].view(p.shape)
                p.data.copy_(new_data)
                idx += numel
        except Exception as e:
            print(f"Error updating parameters: {e}")
            return False
        return True

def train_with_mh(episodes=1000):
    torch.set_grad_enabled(False)  # Disable gradients for stability
    
    try:
        # Initialize environment and policy
        env = SafeBreakoutEnv(render_mode="human")
        state_size = len(env.reset())
        policy = PolicyNetwork(state_size).to('cpu')  # Force CPU for stability
        optimizer = SimplifiedMHOptimizer(policy)
        
        # Get initial parameters
        current_params = torch.cat([p.data.flatten() for p in policy.parameters()])
        current_reward = optimizer.evaluate_policy(env)
        best_params = current_params.clone()
        best_reward = current_reward
        
        # Main training loop
        for episode in range(episodes):
            try:
                # Propose new parameters
                noise = torch.randn_like(current_params) * optimizer.sigma
                proposed_params = current_params + noise
                
                # Evaluate proposed parameters
                if optimizer.update_parameters(proposed_params):
                    proposed_reward = optimizer.evaluate_policy(env)
                    
                    # Accept or reject
                    if proposed_reward > current_reward:
                        current_params = proposed_params.clone()
                        current_reward = proposed_reward
                        
                        if current_reward > best_reward:
                            best_params = current_params.clone()
                            best_reward = current_reward
                            
                            # Save best model
                            torch.save(policy.state_dict(), 'best_model.pt')
                    
                    elif np.random.random() < np.exp((proposed_reward - current_reward) / 10.0):
                        current_params = proposed_params.clone()
                        current_reward = proposed_reward
                
                # Restore current parameters
                optimizer.update_parameters(current_params)
                
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Current Reward: {current_reward:.2f} | "
                      f"Best Reward: {best_reward:.2f}")
                
                # Safety check for catastrophic failure
                if episode > 10 and best_reward < -1000:
                    print("Training appears unstable, stopping...")
                    break
                    
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                break
            except Exception as e:
                print(f"Error in training loop: {e}")
                continue
                
    except Exception as e:
        print(f"Fatal error in training: {e}")
    finally:
        try:
            pygame.quit()
        except:
            pass
        
    return policy