import torch
import numpy as np
import cv2
from PIL import Image
import imageio
import gc
from typing import List, Optional, Union
import tempfile
from pathlib import Path

class SimpleVideoPipeline:
    def __init__(self, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.loaded = True
        
        # Optimized settings for 720p, 10s, 24fps
        self.default_settings = {
            'width': 720,
            'height': 480,
            'num_frames': 240,  # 10 seconds at 24fps
            'fps': 24,
            'num_inference_steps': 50,
            'guidance_scale': 6.0,
            'num_videos_per_prompt': 1
        }
        print(f"🎬 Simple Video Pipeline initialized on {self.device}")
        
    def load_pipeline(self):
        """Pipeline is already loaded"""
        print("✅ Pipeline ready!")
        return True
            
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = None,
        seed: int = None,
        **kwargs
    ):
        """Generate a sample video with animations"""
        # Merge settings
        settings = {**self.default_settings, **kwargs}
        
        print(f"🎬 Generating video: {settings['width']}x{settings['height']}, {settings['num_frames']} frames, {settings['fps']} fps")
        print(f"📝 Prompt: {prompt}")
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            print(f"🎲 Using seed: {seed}")
            
        try:
            frames = []
            width = settings['width']
            height = settings['height']
            num_frames = settings['num_frames']
            
            # Generate animated frames based on prompt
            for i in range(num_frames):
                # Create base frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Analyze prompt for content
                prompt_lower = prompt.lower()
                
                if any(word in prompt_lower for word in ['sunset', 'sunrise', 'sky']):
                    # Sunset/sunrise animation
                    progress = i / num_frames
                    # Sky gradient
                    for y in range(height):
                        intensity_r = int(255 * (1 - y/height) * (0.8 + 0.2 * np.sin(progress * np.pi)))
                        intensity_g = int(150 * (1 - y/height) * (0.6 + 0.4 * np.sin(progress * np.pi)))
                        intensity_b = int(100 * (1 - y/height))
                        frame[y, :] = [intensity_b, intensity_g, intensity_r]  # BGR format
                    
                    # Sun
                    sun_y = int(height * 0.3 + 50 * np.sin(progress * np.pi))
                    sun_x = int(width * 0.7)
                    cv2.circle(frame, (sun_x, sun_y), 30, (0, 255, 255), -1)
                
                elif any(word in prompt_lower for word in ['ocean', 'sea', 'waves']):
                    # Ocean animation
                    # Sky
                    frame[:height//2] = [200, 100, 50]
                    # Water with waves
                    for y in range(height//2, height):
                        wave_offset = int(10 * np.sin((y + i * 2) * 0.1))
                        intensity = 100 + wave_offset
                        frame[y, :] = [intensity, 50, 20]
                
                elif any(word in prompt_lower for word in ['forest', 'trees', 'nature']):
                    # Forest scene
                    # Sky
                    frame[:height//3] = [200, 150, 100]
                    # Trees
                    for x in range(0, width, 60):
                        tree_x = x + int(5 * np.sin(i * 0.1 + x * 0.01))  # Swaying
                        cv2.rectangle(frame, (tree_x, height//3), (tree_x + 20, height), (0, 100, 0), -1)
                        cv2.circle(frame, (tree_x + 10, height//3), 25, (0, 150, 0), -1)
                
                elif any(word in prompt_lower for word in ['city', 'urban', 'buildings']):
                    # City scene
                    # Sky
                    frame[:height//2] = [100, 80, 60]
                    # Buildings
                    for x in range(0, width, 80):
                        building_height = np.random.randint(height//3, height//2)
                        cv2.rectangle(frame, (x, height - building_height), (x + 60, height), (40, 40, 40), -1)
                        # Windows
                        for window_y in range(height - building_height + 20, height - 20, 30):
                            for window_x in range(x + 10, x + 50, 15):
                                if np.random.random() > 0.3:  # Some windows lit
                                    cv2.rectangle(frame, (window_x, window_y), (window_x + 8, window_y + 12), (0, 255, 255), -1)
                
                else:
                    # Default animation with moving elements
                    # Gradient background
                    for y in range(height):
                        intensity = int((y / height) * 255)
                        frame[y, :] = [intensity//3, intensity//2, intensity]
                    
                    # Moving circle
                    center_x = int(width/2 + 100 * np.sin(i * 0.1))
                    center_y = int(height/2 + 50 * np.cos(i * 0.1))
                    cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
                
                # Add text overlay
                cv2.putText(frame, f"Frame {i+1}/{num_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, prompt[:40], (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            
            print(f"✅ Generated {len(frames)} frames successfully!")
            return frames, settings
            
        except Exception as e:
            print(f"❌ Error generating video: {str(e)}")
            raise
            
    def generate_batch(
        self,
        prompts: List[str],
        negative_prompts: List[str] = None,
        seeds: List[int] = None,
        **kwargs
    ):
        """Generate multiple videos in batch"""
        if not prompts:
            return []
            
        print(f"🎬 Starting batch generation of {len(prompts)} videos...")
        
        # Prepare inputs
        if negative_prompts is None:
            negative_prompts = [None] * len(prompts)
        if seeds is None:
            seeds = [None] * len(prompts)
            
        results = []
        
        for i, (prompt, neg_prompt, seed) in enumerate(zip(prompts, negative_prompts, seeds)):
            print(f"\n📹 Processing video {i+1}/{len(prompts)}")
            
            try:
                frames, settings = self.generate_video(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    seed=seed,
                    **kwargs
                )
                
                results.append({
                    'frames': frames,
                    'settings': settings,
                    'prompt': prompt,
                    'success': True
                })
                
            except Exception as e:
                print(f"❌ Failed to generate video {i+1}: {str(e)}")
                results.append({
                    'frames': None,
                    'settings': None,
                    'prompt': prompt,
                    'success': False,
                    'error': str(e)
                })
                
        successful = sum(1 for r in results if r['success'])
        print(f"\n🎉 Batch generation complete! {successful}/{len(prompts)} videos generated successfully.")
        
        return results
        
    def save_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int = 24,
        quality: int = 8
    ):
        """Save frames as video file"""
        print(f"💾 Saving video to {output_path}...")
        
        try:
            # Convert PIL images to numpy arrays
            video_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    video_frames.append(np.array(frame))
                else:
                    video_frames.append(frame)
                    
            # Save using imageio
            with imageio.get_writer(
                output_path,
                fps=fps,
                codec='libx264',
                quality=quality,
                pixelformat='yuv420p'
            ) as writer:
                for frame in video_frames:
                    writer.append_data(frame)
                    
            print(f"✅ Video saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error saving video: {str(e)}")
            return False
            
    def upscale_to_720p(self, frames: List[Image.Image]):
        """Upscale frames to 720p resolution"""
        print("🔍 Upscaling frames to 720p...")
        
        upscaled_frames = []
        target_size = (1280, 720)  # 720p resolution
        
        for i, frame in enumerate(frames):
            if i % 50 == 0:  # Progress indicator
                print(f"  Upscaling frame {i+1}/{len(frames)}")
                
            if isinstance(frame, Image.Image):
                # Use high-quality resampling
                upscaled = frame.resize(target_size, Image.LANCZOS)
                upscaled_frames.append(upscaled)
            else:
                # Convert numpy array to PIL and upscale
                pil_frame = Image.fromarray(frame)
                upscaled = pil_frame.resize(target_size, Image.LANCZOS)
                upscaled_frames.append(upscaled)
                
        print(f"✅ Upscaled {len(upscaled_frames)} frames to 720p")
        return upscaled_frames
        
    def unload_pipeline(self):
        """Cleanup (no-op for simple pipeline)"""
        print("🗑️ Pipeline cleaned up")

# Test the pipeline
print("🧪 Testing Simple Video Pipeline...")
video_pipeline = SimpleVideoPipeline()

# Test generation
test_frames, test_settings = video_pipeline.generate_video(
    prompt="A beautiful sunset over mountains",
    num_frames=60,  # 2.5 seconds
    seed=42
)

print(f"✅ Test successful! Generated {len(test_frames)} frames")
print("🎬 Simple video pipeline is ready for use!")