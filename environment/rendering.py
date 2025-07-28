import pygame
import numpy as np
from typing import Dict, Tuple

class GameRenderer:
    """
    Pygame-based renderer for the Garbage Collection Environment.
    Handles all visual rendering with proper colors and delays.
    """
    
    # Grid cell types (must match custom_env.py)
    EMPTY = 0
    AGENT = 1
    GARBAGE = 2
    HOME = 3
    RIVER = 4
    OBSTACLE = 5
    
    # Color scheme
    COLORS = {
        EMPTY: (255, 255, 255),        # White
        GARBAGE: (255, 0, 0),          # Red circle for garbage
        HOME: (0, 255, 0),             # Green square for home
        RIVER: (0, 0, 139),            # Dark blue square for river
        OBSTACLE: (0, 0, 0),           # Black square for obstacles
    }
    
    # Agent colors
    AGENT_EMPTY = (0, 0, 255)          # Blue square when empty-handed
    AGENT_CARRYING = (255, 0, 0)       # Red square when carrying garbage
    
    def __init__(self, grid_size: int = 12, cell_size: int = 40):
        """
        Initialize the pygame renderer.
        
        Args:
            grid_size: Size of the grid (12x12)
            cell_size: Size of each cell in pixels
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Garbage Collection Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        print(f"Pygame window initialized: {self.window_size}x{self.window_size}")
    
    def render(self, game_state: Dict):
        """
        Render the current game state.
        
        Args:
            game_state: Dictionary containing:
                - grid: 2D numpy array of the game grid
                - agent_pos: (x, y) position of agent
                - carrying_garbage: boolean if agent is carrying garbage
                - mistakes: number of mistakes made
                - disposed_garbage: number of garbage disposed
                - total_garbage: total garbage in episode
        """
        # Handle pygame events (to prevent window from becoming unresponsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        # Clear screen
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw grid
        self._draw_grid(game_state['grid'])
        
        # Draw agent
        self._draw_agent(game_state['agent_pos'], game_state['carrying_garbage'])
        
        # Draw status information
        self._draw_status(game_state)
        
        # Update display
        pygame.display.flip()
        
        # Add delay for visibility (300ms as requested)
        pygame.time.delay(300)
    
    def _draw_grid(self, grid: np.ndarray):
        """Draw the game grid with all entities except the agent."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Calculate pixel position
                pixel_x = y * self.cell_size
                pixel_y = x * self.cell_size
                
                # Create rectangle for this cell
                rect = pygame.Rect(pixel_x, pixel_y, self.cell_size, self.cell_size)
                
                # Get cell type and draw accordingly
                cell_type = grid[x, y]
                
                if cell_type == self.EMPTY:
                    # Draw white square with border
                    pygame.draw.rect(self.screen, self.COLORS[self.EMPTY], rect)
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Light gray border
                
                elif cell_type == self.GARBAGE:
                    # Draw white background then red circle
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
                    center = (pixel_x + self.cell_size // 2, pixel_y + self.cell_size // 2)
                    pygame.draw.circle(self.screen, self.COLORS[self.GARBAGE], center, self.cell_size // 3)
                
                elif cell_type == self.HOME:
                    # Draw green square
                    pygame.draw.rect(self.screen, self.COLORS[self.HOME], rect)
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)  # Black border
                
                elif cell_type == self.RIVER:
                    # Draw dark blue square
                    pygame.draw.rect(self.screen, self.COLORS[self.RIVER], rect)
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)  # Black border
                
                elif cell_type == self.OBSTACLE:
                    # Draw black square
                    pygame.draw.rect(self.screen, self.COLORS[self.OBSTACLE], rect)
    
    def _draw_agent(self, agent_pos: Tuple[int, int], carrying_garbage: bool):
        """
        Draw the agent as a square.
        Blue when empty-handed, red when carrying garbage.
        """
        x, y = agent_pos
        pixel_x = y * self.cell_size
        pixel_y = x * self.cell_size
        
        # Create rectangle for agent
        rect = pygame.Rect(pixel_x, pixel_y, self.cell_size, self.cell_size)
        
        # Choose color based on whether carrying garbage
        if carrying_garbage:
            color = self.AGENT_CARRYING  # Red when carrying
        else:
            color = self.AGENT_EMPTY     # Blue when empty
        
        # Draw agent square
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 3)  # Thick black border
    
    def _draw_status(self, game_state: Dict):
        """Draw status information at the bottom of the window."""
        # Create status text
        status_lines = [
            f"Garbage: {game_state['disposed_garbage']}/{game_state['total_garbage']}",
            f"Mistakes: {game_state['mistakes']}/3",
            f"Carrying: {'Yes' if game_state['carrying_garbage'] else 'No'}"
        ]
        
        # Draw semi-transparent background for text
        status_height = len(status_lines) * 25 + 10
        status_rect = pygame.Rect(0, self.window_size - status_height, self.window_size, status_height)
        status_surface = pygame.Surface((self.window_size, status_height))
        status_surface.set_alpha(200)  # Semi-transparent
        status_surface.fill((50, 50, 50))  # Dark gray
        self.screen.blit(status_surface, (0, self.window_size - status_height))
        
        # Draw text lines
        for i, line in enumerate(status_lines):
            text_surface = self.font.render(line, True, (255, 255, 255))  # White text
            text_y = self.window_size - status_height + 5 + (i * 25)
            self.screen.blit(text_surface, (10, text_y))
    
    def close(self):
        """Close the pygame window and quit."""
        pygame.quit()
        print("Pygame window closed.")


# Test function for the renderer (can be run independently)
def test_renderer():
    """Test the renderer with dummy data."""
    print("Testing renderer...")
    
    renderer = GameRenderer()
    
    # Create dummy game state
    grid = np.zeros((12, 12), dtype=np.int32)
    
    # Add some obstacles
    grid[1, 1] = renderer.OBSTACLE
    grid[2, 3] = renderer.OBSTACLE
    grid[5, 7] = renderer.OBSTACLE
    
    # Add facilities
    grid[0, 0] = renderer.HOME    # Home at top-left
    grid[11, 11] = renderer.RIVER # River at bottom-right
    
    # Add garbage
    grid[3, 4] = renderer.GARBAGE
    grid[7, 8] = renderer.GARBAGE
    
    game_state = {
        'grid': grid,
        'agent_pos': (6, 6),  # Center
        'carrying_garbage': False,
        'mistakes': 1,
        'disposed_garbage': 0,
        'total_garbage': 2
    }
    
    # Render for a few seconds
    import time
    for i in range(10):
        renderer.render(game_state)
        # Move agent around
        game_state['agent_pos'] = ((i + 6) % 12, (i + 6) % 12)
        if i == 5:
            game_state['carrying_garbage'] = True
        time.sleep(0.5)
    
    renderer.close()


if __name__ == "__main__":
    test_renderer()