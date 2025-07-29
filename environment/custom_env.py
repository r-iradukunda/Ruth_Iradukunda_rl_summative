import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
from rendering import GameRenderer

class GarbageCollectionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode="human"):
        super().__init__()
        
        # Grid dimensions
        self.grid_size = 12
        self.render_mode = render_mode
        
        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=PICK_UP, 5=DROP
        self.action_space = spaces.Discrete(6)
        
        # Observation space: grid state + agent position + carrying state
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=4, shape=(self.grid_size, self.grid_size), dtype=np.int32),
            'agent_pos': spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            'carrying': spaces.Discrete(2)  # 0 = not carrying, 1 = carrying
        })
        
        # Game state
        self.grid = None
        self.agent_pos = None
        self.carrying = False
        self.total_reward = 0
        self.garbage_collected = 0
        self.max_garbage = 3
        
        # Grid cell types
        self.EMPTY = 0
        self.OBSTACLE = 1
        self.GARBAGE = 2
        self.HOUSE = 3
        self.RECYCLE_BIN = 4
        
        # Initialize renderer
        self.renderer = None
        if self.render_mode == "human":
            self.renderer = GameRenderer(self.grid_size)
        
        self.reset()
    
    def _get_empty_positions(self):
        """Get all empty positions in the grid"""
        empty_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == self.EMPTY:
                    empty_positions.append((i, j))
        return empty_positions
    
    def _place_obstacles(self):
        """Place random obstacles (black squares) on the grid"""
        num_obstacles = random.randint(10, 20)
        empty_positions = self._get_empty_positions()
        
        # Remove facility positions from possible obstacle positions
        facility_positions = {(0, 0), (0, 11)}
        empty_positions = [pos for pos in empty_positions if pos not in facility_positions]
        
        obstacle_positions = random.sample(empty_positions, min(num_obstacles, len(empty_positions)))
        
        for pos in obstacle_positions:
            self.grid[pos[0]][pos[1]] = self.OBSTACLE
    
    def _place_garbage(self):
        """Place garbage items randomly on the grid"""
        empty_positions = self._get_empty_positions()
        garbage_positions = random.sample(empty_positions, min(self.max_garbage, len(empty_positions)))
        
        for pos in garbage_positions:
            self.grid[pos[0]][pos[1]] = self.GARBAGE
    
    def _place_agent(self):
        """Place agent at random empty position"""
        empty_positions = self._get_empty_positions()
        if empty_positions:
            self.agent_pos = random.choice(empty_positions)
        else:
            self.agent_pos = (1, 1)  # Fallback position
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place facilities at fixed positions
        self.grid[0, 0] = self.HOUSE  # Top-left corner
        self.grid[0, 11] = self.RECYCLE_BIN  # Top-right corner
        
        # Place obstacles
        self._place_obstacles()
        
        # Place garbage
        self._place_garbage()
        
        # Place agent
        self._place_agent()
        
        # Reset game state
        self.carrying = False
        self.total_reward = 0
        self.garbage_collected = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation"""
        return {
            'grid': self.grid.copy(),
            'agent_pos': np.array(self.agent_pos, dtype=np.int32),
            'carrying': int(self.carrying)
        }
    
    def _is_valid_position(self, pos):
        """Check if position is valid and not an obstacle"""
        row, col = pos
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return False
        return self.grid[row, col] != self.OBSTACLE
    
    def step(self, action):
        reward = 0
        terminated = False
        info = {}
        
        # Movement actions
        if action == 0:  # UP
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 1:  # DOWN
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 2:  # LEFT
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == 3:  # RIGHT
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        elif action == 4:  # PICK_UP
            if not self.carrying and self.grid[self.agent_pos[0], self.agent_pos[1]] == self.GARBAGE:
                # Pick up garbage
                self.grid[self.agent_pos[0], self.agent_pos[1]] = self.EMPTY
                self.carrying = True
                reward = 1
                info['action'] = 'picked_up_garbage'
            elif not self.carrying:
                # Try to pick up on empty
                reward = -1
                info['action'] = 'pick_up_failed'
            else:
                # Already carrying
                reward = -0.1
                info['action'] = 'already_carrying'
            new_pos = self.agent_pos
        elif action == 5:  # DROP
            if self.carrying:
                current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
                if current_cell == self.HOUSE:
                    # Drop at house
                    reward = 5
                    self.carrying = False
                    self.garbage_collected += 1
                    info['action'] = 'dropped_at_house'
                elif current_cell == self.RECYCLE_BIN:
                    # Drop at recycling bin
                    reward = 10
                    self.carrying = False
                    self.garbage_collected += 1
                    info['action'] = 'dropped_at_recycle'
                else:
                    # Invalid drop location
                    reward = -1
                    info['action'] = 'invalid_drop'
            else:
                # Not carrying anything
                reward = -1
                info['action'] = 'drop_failed_not_carrying'
            new_pos = self.agent_pos
        
        # Handle movement
        if action in [0, 1, 2, 3]:
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                reward = -0.1  # Small penalty for movement
                info['action'] = 'moved'
            else:
                # Collision with obstacle or boundary
                reward = -1
                info['action'] = 'collision'
        
        # Update total reward
        self.total_reward += reward
        
        # Check termination condition
        if self.garbage_collected >= self.max_garbage:
            terminated = True
            info['final_reward'] = self.total_reward
        
        return self._get_observation(), reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(self.grid, self.agent_pos, self.carrying, self.total_reward)
            return True
        return None
    
    def close(self):
        if hasattr(self, 'renderer') and self.renderer:
            self.renderer.close()

# Main execution with random agent
if __name__ == "__main__":
    # Initialize pygame first
    pygame.init()
    
    env = GarbageCollectionEnv(render_mode="human")
    
    print("Starting Garbage Collection Environment...")
    print("Press Ctrl+C or close the window to exit")
    
    try:
        while True:
            obs, info = env.reset()
            done = False
            step_count = 0
            
            print(f"New episode started. Agent at {env.agent_pos}")
            
            while not done:
                # Handle pygame events first
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                # Random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Render the environment
                env.render()
                
                # Add delay for visibility
                pygame.time.delay(300)
                
                step_count += 1
                
                # Print some debug info occasionally
                if step_count % 20 == 0:
                    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "PICK_UP", "DROP"]
                    print(f"Step {step_count}: Action={action_names[action]}, Reward={reward:.1f}, Total={env.total_reward:.1f}")
                
                if done:
                    print(f"Episode completed! Final reward: {env.total_reward:.1f}")
                    # Show completion message
                    if env.renderer:
                        env.renderer.show_completion(env.total_reward)
                    pygame.time.delay(3000)  # Wait 3 seconds before restarting
                    break
                    
    except KeyboardInterrupt:
        print("Game stopped by user")
    finally:
        env.close()
        pygame.quit()
        print("Game closed.")