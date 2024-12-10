import pygame
import numpy as np
import math
import sys

# Import constants from constants.py
from constants import *

class BreakoutEnv:
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Breakout Policy Gradients")
            self.clock = pygame.time.Clock()
        
        # Handle Pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        self.reset()
    
    def reset(self):
        # Reset paddle to center
        self.paddle_x = SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2
        
        # Reset ball position and velocity
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT - 100
        self.ball_dx = BALL_SPEED * math.sin(ANGLE)
        self.ball_dy = -BALL_SPEED * math.cos(ANGLE)
        
        # Reset game stats
        self.score = 0
        self.steps = 0
        self.game_started = True
        self.first_brick_broken = False
        
        # Recreate all bricks
        self.bricks = []
        for row in range(BRICK_ROWS):
            total_bricks_width = BRICK_COLS * BRICK_WIDTH + (BRICK_COLS - 1) * BRICK_PADDING
            start_x = (SCREEN_WIDTH - total_bricks_width) // 2
            
            for col in range(BRICK_COLS):
                x = start_x + col * (BRICK_WIDTH + BRICK_PADDING)
                y = row * (BRICK_HEIGHT + BRICK_PADDING) + BRICK_PADDING * 2
                self.bricks.append({
                    'rect': pygame.Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT),
                    'color': BRICK_COLORS[row],
                    'active': True
                })
        
        return self.get_state()

    def get_state(self):
        # game_state: 0 for ongoing, 1 for lost, 2 for won
        if self.ball_y > SCREEN_HEIGHT:  # lost
            game_state = 1
        elif not any(brick['active'] for brick in self.bricks):  # won
            game_state = 2
        else:  # ongoing
            game_state = 0
            
        # Normalize coordinates and velocities
        state = [
            game_state,                      # game_state
            self.ball_x / SCREEN_WIDTH,      # ball x position
            self.ball_y / SCREEN_HEIGHT,     # ball y position
            self.ball_dx / BALL_SPEED,       # ball x velocity
            self.ball_dy / BALL_SPEED,       # ball y velocity
            self.paddle_x / SCREEN_WIDTH,    # paddle position
        ]
        
        # Add active brick coordinates
        active_bricks = []
        for brick in self.bricks:
            if brick['active']:
                # Normalize brick coordinates
                brick_x = brick['rect'].x / SCREEN_WIDTH
                brick_y = brick['rect'].y / SCREEN_HEIGHT
                active_bricks.extend([brick_x, brick_y])
        
        # Add time (steps) and score
        state.extend([
            self.steps / MAX_STEPS_PER_EPISODE,  # normalized time
            self.score / (BRICK_ROWS * BRICK_COLS * 10)  # normalized score
        ])
        
        return np.array(state, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0
        done = False
        
        # Calculate new paddle position before clipping
        new_paddle_x = self.paddle_x + action * PADDLE_SPEED
        
        # Check for boundary violations and apply penalties
        boundary_violation = False
        if new_paddle_x < 0:
            boundary_violation = True
            new_paddle_x = 0
            reward -= 2  # Penalty for trying to move out of bounds
        elif new_paddle_x > SCREEN_WIDTH - PADDLE_WIDTH:
            boundary_violation = True
            new_paddle_x = SCREEN_WIDTH - PADDLE_WIDTH
            reward -= 2  # Penalty for trying to move out of bounds
            
        # If we're near the boundary and still trying to move towards it, add extra penalty
        if boundary_violation:
            # Add velocity-based penalty to discourage continuing to push against the wall
            velocity_penalty = abs(action) * 3  # Scales with how hard it's trying to move
            reward -= velocity_penalty
            
        self.paddle_x = new_paddle_x
        
        # Update ball position
        new_ball_x = self.ball_x + self.ball_dx
        new_ball_y = self.ball_y + self.ball_dy
        
        # Wall collisions
        if new_ball_x <= BALL_SIZE or new_ball_x >= SCREEN_WIDTH - BALL_SIZE:
            self.ball_dx = -self.ball_dx
            new_ball_x = self.ball_x + self.ball_dx
        
        if new_ball_y <= BALL_SIZE:
            self.ball_dy = -self.ball_dy
            new_ball_y = self.ball_y + self.ball_dy
        
        # Paddle collision
        paddle_rect = pygame.Rect(self.paddle_x, SCREEN_HEIGHT - 50, PADDLE_WIDTH, PADDLE_HEIGHT)
        if (new_ball_y + BALL_SIZE >= paddle_rect.top and 
            paddle_rect.left <= new_ball_x <= paddle_rect.right):
            
            hit_pos = (new_ball_x - paddle_rect.left) / PADDLE_WIDTH
            angle = (hit_pos - 0.5) * math.pi / 3
            
            speed = math.sqrt(self.ball_dx ** 2 + self.ball_dy ** 2)
            self.ball_dx = speed * math.sin(angle)
            self.ball_dy = -speed * math.cos(angle)
            
            new_ball_y = paddle_rect.top - BALL_SIZE
            reward += 1  # Fixed reward of 1 for any paddle hit
        
        # Brick collisions
        for brick in self.bricks:
            if brick['active'] and brick['rect'].collidepoint(new_ball_x, new_ball_y):
                brick['active'] = False
                self.ball_dy = -self.ball_dy
                
                # Only give reward if it's not the first brick
                if self.first_brick_broken:
                    reward += 10  # Reward for breaking subsequent bricks
                else:
                    self.first_brick_broken = True
                
                self.score += 10
                break
        
        # Game over conditions
        if new_ball_y > SCREEN_HEIGHT:
            done = True
            reward = -50  # Penalty for losing the ball
        
        if not any(brick['active'] for brick in self.bricks):
            done = True
            reward += 100  # Bonus for clearing all bricks
        
        self.ball_x = new_ball_x
        self.ball_y = new_ball_y
        
        if self.render_mode == "human":
            self.render()
        
        return self.get_state(), reward, done
    
    def render(self):
        self.screen.fill((0, 0, 0))
        
        # Draw bricks
        for brick in self.bricks:
            if brick['active']:
                pygame.draw.rect(self.screen, brick['color'], brick['rect'])
        
        # Draw paddle
        pygame.draw.rect(self.screen, WHITE, 
                        (self.paddle_x, SCREEN_HEIGHT - 50, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        # Draw ball
        pygame.draw.circle(self.screen, WHITE, 
                         (int(self.ball_x), int(self.ball_y)), BALL_SIZE)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60)