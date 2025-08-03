# random_agent_demo.py
"""
Static demonstration of random agent in Garbage Collection Environment
This shows the environment visualization without any trained model
"""

import os
import sys
import time
import random
import pygame
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.rendering import GarbageCollectionEnv

# Try to import PIL for GIF creation
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL (Pillow) not available. GIF creation will use alternative method.")

class RandomAgentDemo:
    def __init__(self):
        self.env = GarbageCollectionEnv(render_mode="human")
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROP"]
        self.frames = []  # Store frames for GIF creation
        
    def capture_frame(self):
        """Capture current pygame surface as a frame for GIF"""
        # Get the pygame surface
        surface = pygame.display.get_surface()
        if surface:
            # Convert pygame surface to numpy array
            frame_array = pygame.surfarray.array3d(surface)
            # Transpose to get correct orientation (pygame uses different axis order)
            frame_array = np.transpose(frame_array, (1, 0, 2))
            
            if PIL_AVAILABLE:
                # Convert to PIL Image
                frame_image = Image.fromarray(frame_array)
                self.frames.append(frame_image)
            else:
                # Store as numpy array
                self.frames.append(frame_array)
    
    def save_gif(self, filename=None):
        """Save captured frames as GIF"""
        if not self.frames:
            print("No frames captured to save as GIF")
            return
        
        if filename is None:
            filename = "random_agent_custom_env.gif"
        
        # Ensure the filename has .gif extension
        if not filename.endswith('.gif'):
            filename += '.gif'
        
        # Save in the root directory (parent of demo folder)
        root_dir = os.path.join(os.path.dirname(__file__), '..')
        gif_path = os.path.join(root_dir, filename)
        
        if PIL_AVAILABLE:
            try:
                # Save as animated GIF with PIL
                self.frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.frames[1:],
                    duration=300,  # Duration between frames in milliseconds
                    loop=0  # Loop indefinitely
                )
                print(f"GIF saved successfully: {gif_path}")
                return
            except Exception as e:
                print(f"Error saving GIF with PIL: {e}")
        
        # Fallback methods
        try:
            import imageio
            if PIL_AVAILABLE:
                frame_arrays = [np.array(frame) for frame in self.frames]
            else:
                frame_arrays = self.frames
            imageio.mimsave(gif_path, frame_arrays, duration=0.3)
            print(f"GIF saved with imageio: {gif_path}")
        except ImportError:
            print("Neither PIL nor imageio available for GIF creation")
            # Save frames as individual PNG files instead
            self.save_frames_as_images(os.path.dirname(gif_path), filename.replace('.gif', ''))
        except Exception as e:
            print(f"Error with imageio: {e}")
            self.save_frames_as_images(os.path.dirname(gif_path), filename.replace('.gif', ''))
    
    def save_frames_as_images(self, output_dir, base_filename):
        """Save frames as individual PNG files if GIF creation fails"""
        frames_dir = os.path.join(output_dir, f"{base_filename}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(self.frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            if PIL_AVAILABLE and hasattr(frame, 'save'):
                frame.save(frame_path)
            else:
                # Save numpy array as image using pygame
                frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                pygame.image.save(frame_surface, frame_path)
        
        print(f"Frames saved as individual images in: {frames_dir}")
        print("You can create a GIF manually using online tools or software like GIMP")
        
    def demonstrate_environment(self, num_episodes=3, max_steps_per_episode=100):
        """
        Demonstrate the environment with a random agent
        This shows all possible actions and environment interactions
        """
        print("="*60)
        print("GARBAGE COLLECTION ENVIRONMENT DEMONSTRATION")
        print("="*60)
        print("Environment Details:")
        print(f"- Grid Size: {self.env.grid_size}x{self.env.grid_size}")
        print(f"- Action Space: {self.env.action_space.n} discrete actions")
        print(f"- Observation Space: {self.env.observation_space.shape}")
        print(f"- Actions: {self.action_names}")
        print("="*60)
        
        try:
            for episode in range(num_episodes):
                print(f"\nEPISODE {episode + 1}/{num_episodes}")
                obs, info = self.env.reset()
                
                # Print environment state
                print(f"Agent starting position: {self.env.agent_pos}")
                print(f"House location: {self.env.house_pos}")
                print(f"Recycling facility: {self.env.recycle_pos}")
                print(f"Garbage items: {len(self.env.garbage_positions)}")
                print(f"Carrying garbage: {self.env.carrying}")
                
                episode_reward = 0
                step_count = 0
                actions_taken = {action: 0 for action in self.action_names}
                
                for step in range(max_steps_per_episode):
                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    
                    # Random action selection
                    action = self.env.action_space.sample()
                    action_name = self.action_names[action]
                    actions_taken[action_name] += 1
                    
                    # Execute action
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    # Render environment
                    self.env.render()
                    
                    # Capture frame for GIF (every few frames to reduce file size)
                    if step % 2 == 0:  # Capture every 2nd frame
                        self.capture_frame()
                    
                    # Print action details every 10 steps
                    if step % 10 == 0 or reward != -0.01:  # Show interesting actions
                        print(f"Step {step+1:3d}: {action_name:6s} → Reward: {reward:6.2f} | "
                              f"Carrying: {'Yes' if self.env.carrying else 'No':3s} | "
                              f"Info: {info.get('action', 'none')}")
                    
                    # Check termination
                    if terminated or truncated:
                        print(f"\nEpisode completed at step {step+1}!")
                        if info.get('success', False):
                            print("Mission SUCCESS! All garbage collected and recycled!")
                        break
                    
                    # Add delay for visualization
                    time.sleep(0.3)
                
                # Episode summary
                print(f"\nEPISODE {episode + 1} SUMMARY:")
                print(f"   Total Reward: {episode_reward:.2f}")
                print(f"   Steps Taken: {step_count}")
                print(f"   Garbage Collected: {self.env.garbage_collected}")
                print(f"   Success: {'Yes' if info.get('success', False) else 'No'}")
                print(f"   Action Distribution:")
                for action_name, count in actions_taken.items():
                    percentage = (count / step_count) * 100 if step_count > 0 else 0
                    print(f"     {action_name:8s}: {count:3d} times ({percentage:5.1f}%)")
                
                # Wait between episodes
                if episode < num_episodes - 1:
                    print(f"\nStarting next episode in 3 seconds...")
                    time.sleep(3)

            print(f"\nDEMONSTRATION COMPLETE!")
            print("The random agent explored the environment and demonstrated:")
            print("- All 6 possible actions (UP, DOWN, LEFT, RIGHT, PICKUP, DROP)")
            print("- Environment visualization with pygame")
            print("- Reward system and feedback")
            print("- Termination conditions")
            print("- State transitions and action consequences")
            
            # Save GIF of the demonstration
            print(f"\nSaving GIF with {len(self.frames)} frames...")
            self.save_gif()
            
        except KeyboardInterrupt:
            print("\nDemonstration stopped by user")
            if self.frames:
                print("Saving partial GIF...")
                self.save_gif("random_agent_custom_env_partial.gif")
        finally:
            self.env.close()
            pygame.quit()
    
    def analyze_action_space(self):
        """
        Analyze and document the complete action space
        """
        print("\n" + "="*50)
        print("ACTION SPACE ANALYSIS")
        print("="*50)
        
        action_analysis = {
            0: {
                "name": "UP",
                "description": "Move agent one cell upward",
                "type": "Movement",
                "preconditions": "Valid position (not wall/obstacle)",
                "effects": "Changes agent position, potential collision penalty"
            },
            1: {
                "name": "DOWN", 
                "description": "Move agent one cell downward",
                "type": "Movement",
                "preconditions": "Valid position (not wall/obstacle)",
                "effects": "Changes agent position, potential collision penalty"
            },
            2: {
                "name": "LEFT",
                "description": "Move agent one cell left",
                "type": "Movement", 
                "preconditions": "Valid position (not wall/obstacle)",
                "effects": "Changes agent position, potential collision penalty"
            },
            3: {
                "name": "RIGHT",
                "description": "Move agent one cell right",
                "type": "Movement",
                "preconditions": "Valid position (not wall/obstacle)", 
                "effects": "Changes agent position, potential collision penalty"
            },
            4: {
                "name": "PICKUP",
                "description": "Pick up garbage at current location",
                "type": "Manipulation",
                "preconditions": "Garbage present, not already carrying",
                "effects": "Carrying state = True, positive reward"
            },
            5: {
                "name": "DROP",
                "description": "Drop carried garbage at current location",
                "type": "Manipulation", 
                "preconditions": "Currently carrying garbage",
                "effects": "Carrying state = False, reward based on location"
            }
        }
        
        for action_id, details in action_analysis.items():
            print(f"Action {action_id}: {details['name']}")
            print(f"  Type: {details['type']}")
            print(f"  Description: {details['description']}")
            print(f"  Preconditions: {details['preconditions']}")
            print(f"  Effects: {details['effects']}")
            print()
        
        print("Action Space Properties:")
        print(f"- Type: Discrete")
        print(f"- Size: {self.env.action_space.n}")
        print(f"- Range: 0 to {self.env.action_space.n - 1}")
        print("- All actions are deterministic")
        print("- Actions have different preconditions and effects")
        print("- Invalid actions result in penalty rewards")

def main():
    """Main demonstration function"""
    print("RANDOM AGENT ENVIRONMENT DEMONSTRATION")
    print("This demo shows the environment capabilities without any trained model")
    print("and saves a GIF of the gameplay")
    print("Press Ctrl+C or close the pygame window to stop\n")
    
    demo = RandomAgentDemo()
    
    # First analyze the action space
    demo.analyze_action_space()
    
    # Then demonstrate the environment
    demo.demonstrate_environment(num_episodes=2, max_steps_per_episode=30)

if __name__ == "__main__":
    main()
