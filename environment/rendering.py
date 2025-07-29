import pygame
import os
import sys

class GameRenderer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.cell_size = 50
        self.window_size = grid_size * self.cell_size
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        
        # Initialize Pygame if not already initialized
        if not pygame.get_init():
            pygame.init()
        
        # Create display
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + 60))
        pygame.display.set_caption("Garbage Collection Environment")
        
        # Initialize fonts
        try:
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
        except:
            self.font = pygame.font.SysFont('arial', 36)
            self.small_font = pygame.font.SysFont('arial', 24)
        
        # Load images
        self.images = {}
        self._load_images()
        
        print(f"Renderer initialized. Window size: {self.window_size}x{self.window_size + 60}")
    
    def _load_images(self):
        """Load and scale images from assets folder"""
        image_files = {
            'agent': 'assets/girl.png',
            'garbage': 'assets/water.png',
            'house': 'assets/house.png',
            'recycle': 'assets/recycle-symbol.png'
        }
        
        for key, filepath in image_files.items():
            if os.path.exists(filepath):
                try:
                    image = pygame.image.load(filepath)
                    # Scale image to fit cell size
                    self.images[key] = pygame.transform.scale(image, (self.cell_size - 4, self.cell_size - 4))
                    print(f"Loaded image: {filepath}")
                except pygame.error as e:
                    print(f"Warning: Could not load {filepath}: {e}")
                    self.images[key] = self._create_fallback_image(key)
            else:
                print(f"Warning: Image file {filepath} not found. Using colored rectangle.")
                self.images[key] = self._create_fallback_image(key)
    
    def _create_fallback_image(self, key):
        """Create a colored rectangle as fallback for missing images"""
        fallback = pygame.Surface((self.cell_size - 4, self.cell_size - 4))
        if key == 'agent':
            fallback.fill(self.GREEN)  # Green for agent
        elif key == 'garbage':
            fallback.fill(self.BLUE)  # Blue for garbage
        elif key == 'house':
            fallback.fill(self.RED)  # Red for house
        elif key == 'recycle':
            fallback.fill(self.YELLOW)  # Yellow for recycle
        return fallback
    
    def _draw_grid(self):
        """Draw the grid lines"""
        for i in range(self.grid_size + 1):
            # Vertical lines
            pygame.draw.line(self.screen, self.GRAY, 
                           (i * self.cell_size, 0), 
                           (i * self.cell_size, self.window_size), 1)
            # Horizontal lines
            pygame.draw.line(self.screen, self.GRAY, 
                           (0, i * self.cell_size), 
                           (self.window_size, i * self.cell_size), 1)
    
    def _draw_cell(self, row, col, cell_type):
        """Draw a single cell based on its type"""
        x = col * self.cell_size
        y = row * self.cell_size
        
        # Fill cell background
        cell_rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
        
        if cell_type == 1:  # Obstacle
            pygame.draw.rect(self.screen, self.BLACK, cell_rect)
        else:
            pygame.draw.rect(self.screen, self.WHITE, cell_rect)
            
            # Draw appropriate image based on cell type
            if cell_type == 2:  # Garbage
                self.screen.blit(self.images['garbage'], (x + 2, y + 2))
            elif cell_type == 3:  # House
                self.screen.blit(self.images['house'], (x + 2, y + 2))
            elif cell_type == 4:  # Recycle bin
                self.screen.blit(self.images['recycle'], (x + 2, y + 2))
    
    def _draw_agent(self, agent_pos, carrying):
        """Draw the agent at its position"""
        row, col = agent_pos
        x = col * self.cell_size + 2
        y = row * self.cell_size + 2
        
        # Draw agent image
        self.screen.blit(self.images['agent'], (x, y))
        
        # Draw carrying indicator
        if carrying:
            # Draw a small garbage icon on top of agent
            small_garbage = pygame.transform.scale(self.images['garbage'], (15, 15))
            self.screen.blit(small_garbage, (x + self.cell_size - 20, y))
    
    def _draw_ui(self, total_reward):
        """Draw UI elements (reward counter)"""
        ui_y = self.window_size + 10
        
        # Draw reward
        reward_text = self.font.render(f"Total Reward: {total_reward:.1f}", True, self.BLACK)
        self.screen.blit(reward_text, (10, ui_y))
        
        # Draw instructions
        instruction_text = self.small_font.render("Agent moving randomly - collecting garbage", True, self.GRAY)
        self.screen.blit(instruction_text, (10, ui_y + 30))
    
    def render(self, grid, agent_pos, carrying, total_reward):
        """Main render function"""
        try:
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw grid cells
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    self._draw_cell(row, col, grid[row][col])
            
            # Draw agent
            self._draw_agent(agent_pos, carrying)
            
            # Draw grid lines
            self._draw_grid()
            
            # Draw UI
            self._draw_ui(total_reward)
            
            # Update display
            pygame.display.flip()
            
        except Exception as e:
            print(f"Error in render: {e}")
            import traceback
            traceback.print_exc()
    
    def show_completion(self, final_reward):
        """Show completion message"""
        # Create semi-transparent overlay
        overlay = pygame.Surface((self.window_size, self.window_size))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Draw completion message
        complete_text = self.font.render("Job Complete!", True, self.GREEN)
        reward_text = self.font.render(f"Final Reward: {final_reward:.1f}", True, self.GREEN)
        restart_text = self.small_font.render("Restarting in 3 seconds...", True, self.WHITE)
        
        # Center the text
        complete_rect = complete_text.get_rect(center=(self.window_size // 2, self.window_size // 2 - 40))
        reward_rect = reward_text.get_rect(center=(self.window_size // 2, self.window_size // 2))
        restart_rect = restart_text.get_rect(center=(self.window_size // 2, self.window_size // 2 + 40))
        
        self.screen.blit(complete_text, complete_rect)
        self.screen.blit(reward_text, reward_rect)
        self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def close(self):
        """Clean up pygame resources"""
        try:
            if pygame.get_init():
                pygame.display.quit()
        except:
            pass