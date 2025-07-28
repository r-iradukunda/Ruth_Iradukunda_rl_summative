import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from typing import Tuple, Dict, Optional
from rendering import GameRenderer

class GarbageCollectionEnv(gym.Env):
    """
    Custom Gymnasium environment for grid-based garbage collection game.
    
    The agent must collect garbage pieces and dispose them at the Home facility
    while avoiding obstacles and the River facility.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    # Grid cell types
    EMPTY = 0
    AGENT = 1
    GARBAGE = 2
    HOME = 3
    RIVER = 4
    OBSTACLE = 5
    
    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PICK_UP = 4
    DROP = 5
    
    def __init__(self, grid_size: int = 12, render_mode: Optional[str] = None):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Observation space: flattened grid + agent state
        # Grid cells can be 0-5, agent state includes position and carrying status
        grid_obs_size = grid_size * grid_size
        agent_state_size = 4  # [x, y, carrying_garbage, mistakes]
        
        self.observation_space = spaces.Box(
            low=0, 
            high=max(5, grid_size, 3), 
            shape=(grid_obs_size + agent_state_size,), 
            dtype=np.int32
        )
        
        # Initialize renderer
        self.renderer = None
        if render_mode == "human":
            self.renderer = GameRenderer(grid_size)
        
        # Game state variables
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place obstacles (10-20 blocks)
        num_obstacles = random.randint(10, 20)
        obstacle_positions = set()
        
        while len(obstacle_positions) < num_obstacles:
            x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            obstacle_positions.add((x, y))
        
        for x, y in obstacle_positions:
            self.grid[x, y] = self.OBSTACLE
        
        # Get all empty positions
        empty_positions = [(x, y) for x in range(self.grid_size) 
                          for y in range(self.grid_size) 
                          if self.grid[x, y] == self.EMPTY]
        
        # Place agent at random empty position
        agent_pos = random.choice(empty_positions)
        self.agent_x, self.agent_y = agent_pos
        empty_positions.remove(agent_pos)
        
        # Place Home facility
        home_pos = random.choice(empty_positions)
        self.home_x, self.home_y = home_pos
        self.grid[home_pos[0], home_pos[1]] = self.HOME
        empty_positions.remove(home_pos)
        
        # Place River facility
        river_pos = random.choice(empty_positions)
        self.river_x, self.river_y = river_pos
        self.grid[river_pos[0], river_pos[1]] = self.RIVER
        empty_positions.remove(river_pos)
        
        # Place garbage (3-5 pieces)
        num_garbage = random.randint(3, 5)
        self.garbage_positions = set()
        
        for _ in range(num_garbage):
            if not empty_positions:
                break
            garbage_pos = random.choice(empty_positions)
            self.garbage_positions.add(garbage_pos)
            self.grid[garbage_pos[0], garbage_pos[1]] = self.GARBAGE
            empty_positions.remove(garbage_pos)
        
        # Initialize agent state
        self.carrying_garbage = False
        self.mistakes_count = 0
        self.total_garbage = len(self.garbage_positions)
        self.disposed_garbage = 0
        
        print(f"New episode started! Total garbage to collect: {self.total_garbage}")
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        reward = 0.0
        terminated = False
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK_UP', 'DROP']
        print(f"Action: {action_names[action]}", end=" -> ")
        
        if action in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
            reward = self._move_agent(action)
        elif action == self.PICK_UP:
            reward = self._pick_up_garbage()
        elif action == self.DROP:
            reward = self._drop_garbage()
        
        # Count mistakes (negative rewards)
        if reward < 0:
            self.mistakes_count += 1
            print(f"Reward: {reward:.1f}, Mistakes: {self.mistakes_count}/3")
        else:
            print(f"Reward: {reward:.1f}")
        
        # Check termination conditions
        if self.mistakes_count >= 3:
            terminated = True
            print("Episode ended: Too many mistakes!")
        elif self.disposed_garbage >= self.total_garbage:
            terminated = True
            print("Episode ended: All garbage disposed successfully!")
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _move_agent(self, action: int) -> float:
        """Move the agent and return reward."""
        new_x, new_y = self.agent_x, self.agent_y
        
        if action == self.UP:
            new_x = max(0, self.agent_x - 1)
        elif action == self.DOWN:
            new_x = min(self.grid_size - 1, self.agent_x + 1)
        elif action == self.LEFT:
            new_y = max(0, self.agent_y - 1)
        elif action == self.RIGHT:
            new_y = min(self.grid_size - 1, self.agent_y + 1)
        
        # Check for collision with obstacle
        if self.grid[new_x, new_y] == self.OBSTACLE:
            print("Hit obstacle!", end=" ")
            return -1.0  # Collision penalty
        
        # Valid move
        self.agent_x, self.agent_y = new_x, new_y
        return -0.1  # Small penalty for each move
    
    def _pick_up_garbage(self) -> float:
        """Pick up garbage if present at agent's location."""
        if self.carrying_garbage:
            print("Already carrying garbage!", end=" ")
            return -1.0
        
        agent_pos = (self.agent_x, self.agent_y)
        if agent_pos in self.garbage_positions:
            self.carrying_garbage = True
            self.garbage_positions.remove(agent_pos)
            self.grid[self.agent_x, self.agent_y] = self.EMPTY
            print("Picked up garbage!", end=" ")
            return 1.0
        
        print("No garbage here!", end=" ")
        return -1.0
    
    def _drop_garbage(self) -> float:
        """Drop garbage and return reward based on location."""
        if not self.carrying_garbage:
            print("Not carrying garbage!", end=" ")
            return -1.0
        
        self.carrying_garbage = False
        
        # Check drop location
        if self.agent_x == self.home_x and self.agent_y == self.home_y:
            self.disposed_garbage += 1
            print(f"Disposed at Home! ({self.disposed_garbage}/{self.total_garbage})", end=" ")
            return 10.0
        elif self.agent_x == self.river_x and self.agent_y == self.river_y:
            print("Dropped at River (wrong!)", end=" ")
            return -5.0
        else:
            # Drop at invalid location - put garbage back on grid
            self.grid[self.agent_x, self.agent_y] = self.GARBAGE
            self.garbage_positions.add((self.agent_x, self.agent_y))
            print("Dropped at invalid location!", end=" ")
            return -1.0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as flattened array."""
        # Flatten grid
        grid_flat = self.grid.flatten()
        
        # Agent state: [x, y, carrying_garbage, mistakes]
        agent_state = np.array([
            self.agent_x,
            self.agent_y,
            int(self.carrying_garbage),
            self.mistakes_count
        ], dtype=np.int32)
        
        # Combine grid and agent state
        observation = np.concatenate([grid_flat, agent_state])
        return observation
    
    def _get_info(self) -> Dict:
        """Get info dictionary."""
        return {
            'agent_pos': (self.agent_x, self.agent_y),
            'carrying_garbage': self.carrying_garbage,
            'mistakes': self.mistakes_count,
            'disposed_garbage': self.disposed_garbage,
            'total_garbage': self.total_garbage,
            'garbage_positions': list(self.garbage_positions)
        }
    
    def render(self):
        """Render the environment using pygame."""
        if self.renderer is not None:
            game_state = {
                'grid': self.grid,
                'agent_pos': (self.agent_x, self.agent_y),
                'carrying_garbage': self.carrying_garbage,
                'mistakes': self.mistakes_count,
                'disposed_garbage': self.disposed_garbage,
                'total_garbage': self.total_garbage
            }
            self.renderer.render(game_state)
    
    def close(self):
        """Close the environment."""
        if self.renderer is not None:
            self.renderer.close()


def run_random_agent():
    """Run a random agent in the environment with GUI."""
    print("="*60)
    print("GARBAGE COLLECTION ENVIRONMENT - RANDOM AGENT")
    print("="*60)
    print("🎯 Goal: Collect all garbage and dispose at Home (green)")
    print("❌ Avoid: River (dark blue) and obstacles (black)")
    print("🤖 Agent: Blue square (empty) / Red square (carrying garbage)")
    print("⏰ Each step has a 300ms delay for visibility")
    print("="*60)
    
    # Create environment with rendering
    env = GarbageCollectionEnv(render_mode="human")
    
    try:
        observation, info = env.reset()
        total_reward = 0
        step_count = 0
        
        # Initial render
        env.render()
        
        while True:
            # Random action
            action = env.action_space.sample()
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render
            env.render()
            
            if terminated or truncated:
                break
        
        print(f"\n{'='*60}")
        print("EPISODE SUMMARY:")
        print(f"Steps taken: {step_count}")
        print(f"Total reward: {total_reward:.1f}")
        print(f"Garbage disposed: {info['disposed_garbage']}/{info['total_garbage']}")
        print(f"Mistakes made: {info['mistakes']}")
        
        if info['disposed_garbage'] >= info['total_garbage']:
            print("🎉 SUCCESS: All garbage disposed correctly!")
        else:
            print("❌ FAILED: Too many mistakes")
        
        print("="*60)
        
        # Keep window open for a moment
        import time
        time.sleep(3)
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    run_random_agent()