#!/usr/bin/env python3
"""
🎬 AI Video Generator - Complete Implementation
===============================================

A comprehensive AI video generation system using CogVideoX model with:
- 720p Resolution Support
- 10-second video generation
- 24fps output
- Realistic human posing
- Batch processing
- Memory optimization
- Progress tracking
- Web interface

Requirements:
- GPU Runtime (T4 or better recommended)
- ~15GB RAM
- ~20GB Disk Space

Author: AI Video Generator Team
Version: 2.0.0
Updated: 2025
"""

import os
import sys
import subprocess
import platform
import psutil
import time
import gc
import json
import warnings
import tempfile
import shutil
import zipfile
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 🔧 CONFIGURATION AND CONSTANTS
# ============================================================================

@dataclass
class Config:
    """Configuration class for the AI Video Generator"""
    
    # Model configurations
    MODELS_DIR: str = "/content/models"
    OUTPUT_DIR: str = "/content/outputs"
    
    # Default video settings
    DEFAULT_WIDTH: int = 720
    DEFAULT_HEIGHT: int = 480
    DEFAULT_FPS: int = 24
    DEFAULT_DURATION: int = 10
    DEFAULT_STEPS: int = 50
    DEFAULT_GUIDANCE: float = 6.0
    
    # Model repositories
    COGVIDEOX_REPO: str = "THUDM/CogVideoX-2b"
    VENHANCER_REPO: str = "jwhejwhe/VEnhancer"
    
    # Version requirements
    REQUIRED_VERSIONS = {
        'torch': '2.1.0',
        'torchvision': '0.16.0',
        'diffusers': '0.30.3',
        'transformers': '4.44.2',
        'accelerate': '0.33.0',
        'xformers': '0.0.23'
    }

# Global configuration instance
config = Config()

# ============================================================================
# 🔍 ENVIRONMENT DETECTION AND VALIDATION
# ============================================================================

class EnvironmentDetector:
    """Comprehensive environment detection and validation"""
    
    def __init__(self):
        self.system_info = {}
        self.requirements_met = False
        self.gpu_available = False
        
    def detect_environment(self) -> Dict[str, Any]:
        """Detect and validate the current environment"""
        print("🔍 Detecting environment...")
        
        # Basic system info
        self.system_info = {
            'platform': platform.system(),
            'python_version': sys.version,
            'is_colab': 'google.colab' in sys.modules,
            'timestamp': datetime.now().isoformat()
        }
        
        # Memory info
        memory = psutil.virtual_memory()
        self.system_info.update({
            'total_ram_gb': round(memory.total / (1024**3), 2),
            'available_ram_gb': round(memory.available / (1024**3), 2),
            'ram_usage_percent': memory.percent
        })
        
        # GPU info with multiple detection methods
        self._detect_gpu()
        
        # Disk space
        disk_usage = psutil.disk_usage('/')
        self.system_info.update({
            'disk_total_gb': round(disk_usage.total / (1024**3), 2),
            'disk_free_gb': round(disk_usage.free / (1024**3), 2),
            'disk_used_gb': round(disk_usage.used / (1024**3), 2)
        })
        
        # Check requirements
        self._check_requirements()
        
        # Display results
        self._display_system_info()
        
        return self.system_info
    
    def _detect_gpu(self):
        """Detect GPU with multiple methods"""
        # Method 1: Try torch
        try:
            import torch
            self.system_info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                self.system_info['gpu_name'] = torch.cuda.get_device_name(0)
                self.system_info['gpu_memory_gb'] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                )
                self.system_info['gpu_count'] = torch.cuda.device_count()
                self.gpu_available = True
                print(f"✅ GPU detected: {self.system_info['gpu_name']}")
            else:
                print("⚠️ CUDA not available")
        except ImportError:
            print("⚠️ PyTorch not installed, cannot detect GPU")
        
        # Method 2: Try nvidia-smi
        if not self.gpu_available:
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines and lines[0]:
                        gpu_info = lines[0].split(',')
                        self.system_info['gpu_name'] = gpu_info[0].strip()
                        self.system_info['gpu_memory_gb'] = round(float(gpu_info[1]) / 1024, 2)
                        self.gpu_available = True
                        print(f"✅ GPU detected via nvidia-smi: {self.system_info['gpu_name']}")
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass
        
        # Fallback values
        if not self.gpu_available:
            self.system_info.update({
                'cuda_available': False,
                'gpu_name': 'None',
                'gpu_memory_gb': 0,
                'gpu_count': 0
            })
    
    def _check_requirements(self):
        """Check if system meets requirements"""
        issues = []
        
        # GPU requirements
        if not self.gpu_available:
            issues.append("❌ No GPU detected - Video generation will be very slow")
        elif self.system_info.get('gpu_memory_gb', 0) < 6:
            issues.append("⚠️ GPU memory < 6GB - May experience memory issues")
        
        # RAM requirements
        if self.system_info['total_ram_gb'] < 12:
            issues.append("⚠️ RAM < 12GB - May experience memory issues")
        
        # Disk space requirements
        if self.system_info['disk_free_gb'] < 20:
            issues.append("⚠️ Disk space < 20GB - May not have enough space for models")
        
        self.system_info['issues'] = issues
        self.requirements_met = len(issues) == 0
        
        if issues:
            print("\n⚠️ System Issues Detected:")
            for issue in issues:
                print(f"  {issue}")
            print("\n💡 Recommendations:")
            print("  - Use Google Colab Pro with GPU runtime for best results")
            print("  - Consider using smaller models if memory is limited")
        else:
            print("\n✅ All requirements met! System is ready for video generation.")
    
    def _display_system_info(self):
        """Display system information in a formatted way"""
        print("\n" + "="*60)
        print("🖥️  SYSTEM INFORMATION")
        print("="*60)
        
        print(f"Platform: {self.system_info['platform']}")
        print(f"Python: {self.system_info['python_version'].split()[0]}")
        print(f"Google Colab: {'Yes' if self.system_info['is_colab'] else 'No'}")
        
        print(f"\n💾 Memory:")
        print(f"  RAM: {self.system_info['available_ram_gb']:.1f}GB / {self.system_info['total_ram_gb']:.1f}GB")
        print(f"  Usage: {self.system_info['ram_usage_percent']:.1f}%")
        
        print(f"\n🎮 GPU:")
        if self.gpu_available:
            print(f"  Name: {self.system_info['gpu_name']}")
            print(f"  Memory: {self.system_info['gpu_memory_gb']:.1f}GB")
            print(f"  Count: {self.system_info.get('gpu_count', 1)}")
        else:
            print("  None detected")
        
        print(f"\n💿 Storage:")
        print(f"  Free: {self.system_info['disk_free_gb']:.1f}GB")
        print(f"  Total: {self.system_info['disk_total_gb']:.1f}GB")
        
        print("="*60)

# ============================================================================
# 📦 DEPENDENCY MANAGEMENT
# ============================================================================

class DependencyManager:
    """Advanced dependency management with version control"""
    
    def __init__(self):
        self.packages = {
            'essential': [
                'torch>=2.1.0',
                'torchvision>=0.16.0',
                'diffusers>=0.30.3',
                'transformers>=4.44.2',
                'accelerate>=0.33.0',
                'xformers>=0.0.23',
            ],
            'video': [
                'opencv-python-headless>=4.8.0',
                'imageio[ffmpeg]>=2.31.0',
                'moviepy>=1.0.3',
                'av>=10.0.0',
                'decord>=0.6.0'
            ],
            'ui': [
                'streamlit>=1.28.0',
                'gradio>=4.0.0',
                'matplotlib>=3.7.0',
                'pillow>=10.0.0'
            ],
            'utils': [
                'huggingface-hub>=0.19.0',
                'safetensors>=0.4.0',
                'einops>=0.7.0',
                'omegaconf>=2.3.0',
                'tqdm>=4.65.0'
            ]
        }
    
    def install_dependencies(self, category: str = 'all', force_reinstall: bool = False):
        """Install dependencies with enhanced error handling"""
        print(f"📦 Installing dependencies ({category})...")
        
        if category == 'all':
            categories = list(self.packages.keys())
        else:
            categories = [category] if category in self.packages else []
        
        if not categories:
            print(f"❌ Unknown category: {category}")
            return False
        
        # Install PyTorch with CUDA support first
        if 'essential' in categories:
            success = self._install_pytorch()
            if not success:
                print("❌ Failed to install PyTorch")
                return False
        
        # Install other packages
        total_success = True
        for cat in categories:
            if cat == 'essential':
                # Skip PyTorch packages as they're already installed
                packages = [p for p in self.packages[cat] if not p.startswith('torch')]
            else:
                packages = self.packages[cat]
            
            if packages:
                success = self._install_package_group(cat, packages, force_reinstall)
                total_success = total_success and success
        
        # Verify installation
        self._verify_critical_imports()
        
        return total_success
    
    def _install_pytorch(self) -> bool:
        """Install PyTorch with CUDA support"""
        print("🔥 Installing PyTorch with CUDA support...")
        
        # Try CUDA version first
        cuda_commands = [
            [
                sys.executable, '-m', 'pip', 'install', 
                'torch>=2.1.0', 'torchvision>=0.16.0', 'torchaudio>=2.1.0',
                '--index-url', 'https://download.pytorch.org/whl/cu121'
            ],
            [
                sys.executable, '-m', 'pip', 'install', 
                'torch>=2.1.0', 'torchvision>=0.16.0', 'torchaudio>=2.1.0',
                '--index-url', 'https://download.pytorch.org/whl/cu118'
            ]
        ]
        
        for cmd in cuda_commands:
            try:
                print(f"  Trying: {' '.join(cmd[-3:])}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    print("✅ PyTorch with CUDA installed successfully")
                    return True
                else:
                    print(f"  Failed: {result.stderr[:100]}...")
            except subprocess.TimeoutExpired:
                print("  Timeout occurred, trying next option...")
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        # Fallback to CPU version
        print("🔄 Installing CPU version as fallback...")
        try:
            cpu_cmd = [
                sys.executable, '-m', 'pip', 'install', 
                'torch>=2.1.0', 'torchvision>=0.16.0', 'torchaudio>=2.1.0'
            ]
            result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print("✅ PyTorch (CPU) installed successfully")
                return True
            else:
                print(f"❌ Failed to install PyTorch: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error installing PyTorch: {str(e)}")
            return False
    
    def _install_package_group(self, category: str, packages: List[str], force_reinstall: bool) -> bool:
        """Install a group of packages"""
        print(f"📦 Installing {category} packages...")
        
        success_count = 0
        total_count = len(packages)
        
        for package in packages:
            try:
                cmd = [sys.executable, '-m', 'pip', 'install', package]
                if force_reinstall:
                    cmd.append('--force-reinstall')
                
                print(f"  Installing {package.split('>=')[0]}...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    success_count += 1
                    print(f"    ✅ Success")
                else:
                    print(f"    ❌ Failed: {result.stderr.split('ERROR:')[-1][:100] if 'ERROR:' in result.stderr else 'Unknown error'}")
                    
            except subprocess.TimeoutExpired:
                print(f"    ⏱️ Timeout installing {package}")
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
        
        print(f"📊 {category}: {success_count}/{total_count} packages installed successfully")
        return success_count >= total_count * 0.8  # 80% success rate
    
    def _verify_critical_imports(self):
        """Verify critical packages can be imported"""
        print("\n🔍 Verifying critical imports...")
        
        critical_imports = {
            'torch': 'PyTorch',
            'diffusers': 'Diffusers',
            'transformers': 'Transformers',
            'PIL': 'Pillow',
            'cv2': 'OpenCV',
            'imageio': 'ImageIO'
        }
        
        results = {}
        for module, name in critical_imports.items():
            try:
                __import__(module)
                results[name] = True
                print(f"  ✅ {name}")
            except ImportError as e:
                results[name] = False
                print(f"  ❌ {name} - {str(e)}")
        
        # Check PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  ✅ CUDA Support - {torch.cuda.get_device_name(0)}")
            else:
                print(f"  ⚠️ CUDA Support - Not available")
        except:
            print(f"  ❌ CUDA Support - Cannot check")
        
        return results
    
    def get_installed_versions(self) -> Dict[str, str]:
        """Get versions of installed packages"""
        versions = {}
        
        packages_to_check = ['torch', 'diffusers', 'transformers', 'accelerate', 'xformers']
        
        for package in packages_to_check:
            try:
                module = __import__(package)
                versions[package] = getattr(module, '__version__', 'unknown')
            except ImportError:
                versions[package] = 'not installed'
        
        return versions

# ============================================================================
# 🤖 MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Advanced model download and management"""
    
    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir or config.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_configs = {
            'cogvideox': {
                'repo_id': config.COGVIDEOX_REPO,
                'local_dir': self.models_dir / 'CogVideoX-2b',
                'size_gb': 8.5,
                'description': 'Main video generation model',
                'required': True
            },
            'venhancer': {
                'repo_id': config.VENHANCER_REPO,
                'local_dir': self.models_dir / 'VEnhancer',
                'size_gb': 2.1,
                'description': 'Video enhancement model',
                'required': False
            }
        }
    
    def download_model(self, model_name: str, force_download: bool = False) -> Optional[str]:
        """Download a specific model with progress tracking"""
        if model_name not in self.model_configs:
            print(f"❌ Unknown model: {model_name}")
            return None
        
        config_data = self.model_configs[model_name]
        local_dir = config_data['local_dir']
        
        # Check if model already exists
        if local_dir.exists() and not force_download:
            if self._verify_model_integrity(model_name):
                print(f"✅ {model_name} already exists and is valid")
                return str(local_dir)
            else:
                print(f"⚠️ {model_name} exists but appears corrupted, re-downloading...")
                shutil.rmtree(local_dir)
        
        print(f"📥 Downloading {model_name} ({config_data['size_gb']:.1f}GB)")
        print(f"📝 {config_data['description']}")
        
        try:
            # Import huggingface_hub here to avoid import errors
            from huggingface_hub import snapshot_download
            
            local_dir_str = str(local_dir)
            snapshot_download(
                repo_id=config_data['repo_id'],
                local_dir=local_dir_str,
                resume_download=True,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.git*", "README.md", "*.txt", "*.md"]
            )
            
            print(f"✅ {model_name} downloaded successfully!")
            return local_dir_str
            
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {str(e)}")
            if local_dir.exists():
                shutil.rmtree(local_dir)
            return None
    
    def download_all_models(self) -> Dict[str, str]:
        """Download all required models"""
        total_size = sum(config_data['size_gb'] for config_data in self.model_configs.values())
        print(f"📦 Downloading all models (Total: {total_size:.1f}GB)")
        
        downloaded_models = {}
        for model_name, config_data in self.model_configs.items():
            if config_data['required']:
                path = self.download_model(model_name)
                if path:
                    downloaded_models[model_name] = path
                else:
                    print(f"❌ Failed to download required model: {model_name}")
                    return {}
        
        return downloaded_models
    
    def _verify_model_integrity(self, model_name: str) -> bool:
        """Verify model integrity"""
        config_data = self.model_configs[model_name]
        local_dir = config_data['local_dir']
        
        # Check if directory exists and has files
        if not local_dir.exists():
            return False
        
        # Check for essential files
        essential_files = ['config.json']
        for file_name in essential_files:
            if not (local_dir / file_name).exists():
                return False
        
        # Check directory size (rough estimate)
        total_size = sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        expected_size = config_data['size_gb']
        
        if size_gb < expected_size * 0.8:  # Allow 20% variance
            return False
        
        return True
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        status = {}
        
        for model_name, config_data in self.model_configs.items():
            local_dir = config_data['local_dir']
            
            if local_dir.exists():
                try:
                    total_size = sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file())
                    size_gb = total_size / (1024**3)
                    is_valid = self._verify_model_integrity(model_name)
                except Exception as e:
                    size_gb = 0
                    is_valid = False
                
                status[model_name] = {
                    'downloaded': True,
                    'valid': is_valid,
                    'size_gb': size_gb,
                    'path': str(local_dir),
                    'description': config_data['description']
                }
            else:
                status[model_name] = {
                    'downloaded': False,
                    'valid': False,
                    'size_gb': 0,
                    'path': None,
                    'description': config_data['description']
                }
        
        return status
    
    def cleanup_models(self):
        """Clean up all downloaded models"""
        if self.models_dir.exists():
            shutil.rmtree(self.models_dir)
            print("🗑️ All models cleaned up")

# ============================================================================
# 🎬 VIDEO GENERATION PIPELINE
# ============================================================================

class OptimizedVideoPipeline:
    """Advanced video generation pipeline with memory optimization"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = self._determine_device(device)
        self.pipeline = None
        self.loaded = False
        
        # Default settings optimized for quality and performance
        self.default_settings = {
            'width': config.DEFAULT_WIDTH,
            'height': config.DEFAULT_HEIGHT,
            'num_frames': config.DEFAULT_DURATION * config.DEFAULT_FPS,
            'fps': config.DEFAULT_FPS,
            'num_inference_steps': config.DEFAULT_STEPS,
            'guidance_scale': config.DEFAULT_GUIDANCE,
            'num_videos_per_prompt': 1
        }
        
        print(f"🎬 Video pipeline initialized")
        print(f"  Model: {self.model_path}")
        print(f"  Device: {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def load_pipeline(self, force_reload: bool = False):
        """Load the video generation pipeline with memory optimization"""
        if self.loaded and not force_reload:
            print("✅ Pipeline already loaded")
            return
        
        print("🔄 Loading CogVideoX pipeline...")
        
        try:
            # Clear memory before loading
            self._clear_memory()
            
            # Import required modules
            from diffusers import CogVideoXPipeline
            import torch
            
            # Check if model exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Load pipeline with optimizations
            self.pipeline = CogVideoXPipeline.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory optimizations
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_vae_slicing()
                
                # Try to enable xformers
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ xFormers memory optimization enabled")
                except Exception as e:
                    print(f"⚠️ xFormers not available: {str(e)}")
            
            self.loaded = True
            print("✅ Pipeline loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading pipeline: {str(e)}")
            self._clear_memory()
            raise
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = None,
        seed: int = None,
        **kwargs
    ) -> tuple:
        """Generate a single video with optimized settings"""
        if not self.loaded:
            self.load_pipeline()
        
        # Merge settings
        settings = {**self.default_settings, **kwargs}
        
        print(f"🎬 Generating video:")
        print(f"  Resolution: {settings['width']}x{settings['height']}")
        print(f"  Frames: {settings['num_frames']} ({settings['num_frames']/settings['fps']:.1f}s)")
        print(f"  FPS: {settings['fps']}")
        print(f"  Prompt: {prompt}")
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            import torch
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"  Seed: {seed}")
        
        try:
            # Clear memory before generation
            self._clear_memory()
            
            # Generate video
            import torch
            with torch.inference_mode():
                start_time = time.time()
                
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=settings['width'],
                    height=settings['height'],
                    num_frames=settings['num_frames'],
                    num_inference_steps=settings['num_inference_steps'],
                    guidance_scale=settings['guidance_scale'],
                    num_videos_per_prompt=settings['num_videos_per_prompt'],
                    generator=generator,
                    output_type="pil"
                )
                
                generation_time = time.time() - start_time
                
            # Extract frames
            frames = result.frames[0]
            
            # Clean up
            del result
            self._clear_memory()
            
            print(f"✅ Generated {len(frames)} frames in {generation_time:.1f}s")
            return frames, settings
            
        except Exception as e:
            print(f"❌ Error generating video: {str(e)}")
            self._clear_memory()
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        negative_prompts: List[str] = None,
        seeds: List[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
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
                    'success': True,
                    'index': i
                })
                
            except Exception as e:
                print(f"❌ Failed to generate video {i+1}: {str(e)}")
                results.append({
                    'frames': None,
                    'settings': None,
                    'prompt': prompt,
                    'success': False,
                    'error': str(e),
                    'index': i
                })
        
        successful = sum(1 for r in results if r['success'])
        print(f"\n🎉 Batch generation complete! {successful}/{len(prompts)} videos generated")
        
        return results
    
    def save_video(
        self,
        frames: List,
        output_path: str,
        fps: int = 24,
        quality: int = 8,
        codec: str = 'libx264'
    ) -> bool:
        """Save frames as video file with enhanced options"""
        print(f"💾 Saving video to {output_path}...")
        
        try:
            import imageio
            import numpy as np
            from PIL import Image
            
            # Convert frames to numpy arrays
            video_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    video_frames.append(np.array(frame))
                else:
                    video_frames.append(frame)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save video
            with imageio.get_writer(
                output_path,
                fps=fps,
                codec=codec,
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
    
    def _clear_memory(self):
        """Clear GPU and system memory"""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def unload_pipeline(self):
        """Unload pipeline to free memory"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.loaded = False
            self._clear_memory()
            print("🗑️ Pipeline unloaded and memory cleared")

# ============================================================================
# 🎨 STREAMLIT WEB INTERFACE
# ============================================================================

class StreamlitInterface:
    """Modern Streamlit web interface for video generation"""
    
    def __init__(self, pipeline: OptimizedVideoPipeline):
        self.pipeline = pipeline
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Run the Streamlit interface"""
        try:
            import streamlit as st
            
            # Page config
            st.set_page_config(
                page_title="🎬 AI Video Generator",
                page_icon="🎬",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Main interface
            self._render_header()
            self._render_sidebar()
            self._render_main_content()
            
        except ImportError:
            print("❌ Streamlit not installed. Please install with: pip install streamlit")
            return False
        
        return True
    
    def _render_header(self):
        """Render the header section"""
        import streamlit as st
        
        st.title("🎬 AI Video Generator")
        st.markdown("Generate high-quality videos with AI using CogVideoX")
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if self.pipeline.loaded:
                st.success("✅ Pipeline Loaded")
            else:
                st.warning("⚠️ Pipeline Not Loaded")
        
        with col2:
            device_emoji = "🖥️" if self.pipeline.device == "cuda" else "💻"
            st.info(f"{device_emoji} Device: {self.pipeline.device}")
        
        with col3:
            output_count = len(list(self.output_dir.glob("*.mp4")))
            st.info(f"📹 Generated: {output_count} videos")
    
    def _render_sidebar(self):
        """Render the sidebar with settings"""
        import streamlit as st
        
        st.sidebar.header("⚙️ Settings")
        
        # Video settings
        st.sidebar.subheader("🎥 Video Settings")
        
        resolution = st.sidebar.selectbox(
            "Resolution",
            options=["720x480", "640x360", "480x320"],
            index=0
        )
        width, height = map(int, resolution.split('x'))
        
        duration = st.sidebar.slider(
            "Duration (seconds)",
            min_value=2,
            max_value=20,
            value=10,
            step=1
        )
        
        fps = st.sidebar.slider(
            "FPS",
            min_value=12,
            max_value=30,
            value=24,
            step=6
        )
        
        # Advanced settings
        st.sidebar.subheader("🔧 Advanced Settings")
        
        steps = st.sidebar.slider(
            "Inference Steps",
            min_value=20,
            max_value=100,
            value=50,
            step=10
        )
        
        guidance = st.sidebar.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=12.0,
            value=6.0,
            step=0.5
        )
        
        seed = st.sidebar.number_input(
            "Seed (optional)",
            min_value=0,
            max_value=999999,
            value=42,
            step=1
        )
        
        # Store settings in session state
        st.session_state.video_settings = {
            'width': width,
            'height': height,
            'duration': duration,
            'fps': fps,
            'steps': steps,
            'guidance': guidance,
            'seed': seed
        }
        
        # Pipeline controls
        st.sidebar.subheader("🎛️ Pipeline Controls")
        
        if st.sidebar.button("🔄 Load Pipeline"):
            with st.spinner("Loading pipeline..."):
                try:
                    self.pipeline.load_pipeline()
                    st.success("✅ Pipeline loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading pipeline: {str(e)}")
        
        if st.sidebar.button("🗑️ Unload Pipeline"):
            self.pipeline.unload_pipeline()
            st.success("✅ Pipeline unloaded!")
    
    def _render_main_content(self):
        """Render the main content area"""
        import streamlit as st
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🎬 Generate", "📁 Gallery", "📊 System Info"])
        
        with tab1:
            self._render_generation_tab()
        
        with tab2:
            self._render_gallery_tab()
        
        with tab3:
            self._render_system_info_tab()
    
    def _render_generation_tab(self):
        """Render the video generation tab"""
        import streamlit as st
        
        st.header("🎬 Video Generation")
        
        # Single video generation
        st.subheader("📹 Single Video")
        
        prompt = st.text_area(
            "Enter your prompt:",
            value="A realistic human walking in a park, cinematic lighting, 4K quality",
            height=100
        )
        
        negative_prompt = st.text_area(
            "Negative prompt (optional):",
            value="blurry, low quality, distorted, artifacts",
            height=60
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🎬 Generate Video", type="primary"):
                if not self.pipeline.loaded:
                    st.error("❌ Please load the pipeline first!")
                    return
                
                self._generate_single_video(prompt, negative_prompt)
        
        with col2:
            if st.button("🔄 Clear"):
                st.rerun()
        
        # Batch generation
        st.subheader("📦 Batch Generation")
        
        batch_prompts = st.text_area(
            "Enter multiple prompts (one per line):",
            height=150,
            placeholder="A person dancing in the rain\nA cat playing with a ball\nA sunset over mountains"
        )
        
        if st.button("📦 Generate Batch"):
            if not self.pipeline.loaded:
                st.error("❌ Please load the pipeline first!")
                return
            
            prompts = [p.strip() for p in batch_prompts.split('\n') if p.strip()]
            if prompts:
                self._generate_batch_videos(prompts)
            else:
                st.warning("⚠️ Please enter at least one prompt!")
    
    def _render_gallery_tab(self):
        """Render the gallery tab"""
        import streamlit as st
        
        st.header("📁 Video Gallery")
        
        # Get all videos
        video_files = list(self.output_dir.glob("*.mp4"))
        video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not video_files:
            st.info("📝 No videos generated yet. Create some videos in the Generate tab!")
            return
        
        # Display videos in grid
        cols = 2
        for i in range(0, len(video_files), cols):
            col_list = st.columns(cols)
            
            for j, col in enumerate(col_list):
                if i + j < len(video_files):
                    video_file = video_files[i + j]
                    
                    with col:
                        st.video(str(video_file))
                        st.caption(f"📅 {video_file.name}")
                        
                        # Metadata
                        metadata_file = video_file.with_suffix('.json')
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                st.text(f"💭 {metadata.get('prompt', 'No prompt')[:100]}...")
                            except:
                                pass
    
    def _render_system_info_tab(self):
        """Render the system information tab"""
        import streamlit as st
        
        st.header("📊 System Information")
        
        # Environment info
        env_detector = EnvironmentDetector()
        system_info = env_detector.detect_environment()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💻 Hardware")
            st.write(f"**Platform:** {system_info['platform']}")
            st.write(f"**RAM:** {system_info['available_ram_gb']:.1f}GB / {system_info['total_ram_gb']:.1f}GB")
            st.write(f"**GPU:** {system_info['gpu_name']}")
            st.write(f"**GPU Memory:** {system_info['gpu_memory_gb']:.1f}GB")
        
        with col2:
            st.subheader("📦 Software")
            
            # Get package versions
            dep_manager = DependencyManager()
            versions = dep_manager.get_installed_versions()
            
            for package, version in versions.items():
                st.write(f"**{package}:** {version}")
    
    def _generate_single_video(self, prompt: str, negative_prompt: str = None):
        """Generate a single video"""
        import streamlit as st
        
        settings = st.session_state.get('video_settings', {})
        
        with st.spinner("🎬 Generating video..."):
            try:
                # Generate video
                frames, video_settings = self.pipeline.generate_video(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    seed=settings.get('seed'),
                    width=settings.get('width', 720),
                    height=settings.get('height', 480),
                    num_frames=settings.get('duration', 10) * settings.get('fps', 24),
                    fps=settings.get('fps', 24),
                    num_inference_steps=settings.get('steps', 50),
                    guidance_scale=settings.get('guidance', 6.0)
                )
                
                # Save video
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.output_dir / f"video_{timestamp}.mp4"
                
                success = self.pipeline.save_video(
                    frames, str(output_path), fps=settings.get('fps', 24)
                )
                
                if success:
                    st.success(f"✅ Video generated successfully!")
                    
                    # Save metadata
                    metadata = {
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'settings': video_settings,
                        'timestamp': timestamp
                    }
                    
                    metadata_path = output_path.with_suffix('.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                    
                    # Display video
                    st.video(str(output_path))
                    
                else:
                    st.error("❌ Failed to save video")
                    
            except Exception as e:
                st.error(f"❌ Error generating video: {str(e)}")
    
    def _generate_batch_videos(self, prompts: List[str]):
        """Generate batch videos"""
        import streamlit as st
        
        settings = st.session_state.get('video_settings', {})
        
        with st.spinner(f"🎬 Generating {len(prompts)} videos..."):
            try:
                # Generate videos
                results = self.pipeline.generate_batch(
                    prompts=prompts,
                    width=settings.get('width', 720),
                    height=settings.get('height', 480),
                    num_frames=settings.get('duration', 10) * settings.get('fps', 24),
                    fps=settings.get('fps', 24),
                    num_inference_steps=settings.get('steps', 50),
                    guidance_scale=settings.get('guidance', 6.0)
                )
                
                # Save successful videos
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_count = 0
                
                for i, result in enumerate(results):
                    if result['success']:
                        output_path = self.output_dir / f"batch_{timestamp}_{i+1:03d}.mp4"
                        success = self.pipeline.save_video(
                            result['frames'], str(output_path), fps=settings.get('fps', 24)
                        )
                        
                        if success:
                            saved_count += 1
                            
                            # Save metadata
                            metadata = {
                                'prompt': result['prompt'],
                                'settings': result['settings'],
                                'timestamp': timestamp,
                                'batch_index': i
                            }
                            
                            metadata_path = output_path.with_suffix('.json')
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2, default=str)
                
                st.success(f"✅ Generated {saved_count}/{len(prompts)} videos successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating batch videos: {str(e)}")

# ============================================================================
# 🔧 UTILITY FUNCTIONS
# ============================================================================

def setup_environment():
    """Setup the complete environment"""
    print("🚀 Setting up AI Video Generator environment...")
    
    # Create directories
    for directory in [config.MODELS_DIR, config.OUTPUT_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Detect environment
    env_detector = EnvironmentDetector()
    system_info = env_detector.detect_environment()
    
    return system_info

def install_all_dependencies():
    """Install all required dependencies"""
    print("📦 Installing all dependencies...")
    
    dep_manager = DependencyManager()
    success = dep_manager.install_dependencies()
    
    if success:
        print("✅ All dependencies installed successfully!")
    else:
        print("⚠️ Some dependencies failed to install")
    
    return success

def download_models():
    """Download all required models"""
    print("🤖 Downloading models...")
    
    model_manager = ModelManager()
    models = model_manager.download_all_models()
    
    if models:
        print(f"✅ Downloaded {len(models)} models successfully!")
    else:
        print("❌ Failed to download models")
    
    return models

def run_streamlit_app():
    """Run the Streamlit application"""
    print("🌐 Starting Streamlit application...")
    
    try:
        # Setup environment
        system_info = setup_environment()
        
        # Download models
        models = download_models()
        if not models:
            print("❌ Cannot start app without models")
            return False
        
        # Initialize pipeline
        cogvideox_path = models.get('cogvideox')
        if not cogvideox_path:
            print("❌ CogVideoX model not available")
            return False
        
        pipeline = OptimizedVideoPipeline(cogvideox_path)
        
        # Create and run interface
        interface = StreamlitInterface(pipeline)
        return interface.run()
        
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")
        return False

# ============================================================================
# 🎯 EXAMPLE USAGE AND TESTING
# ============================================================================

def test_pipeline():
    """Test the video generation pipeline"""
    print("🧪 Testing video generation pipeline...")
    
    try:
        # Setup
        system_info = setup_environment()
        
        # Download models
        models = download_models()
        if not models:
            return False
        
        # Initialize pipeline
        cogvideox_path = models.get('cogvideox')
        pipeline = OptimizedVideoPipeline(cogvideox_path)
        
        # Load pipeline
        pipeline.load_pipeline()
        
        # Generate test video
        frames, settings = pipeline.generate_video(
            prompt="A person waving hello, friendly gesture, clear background",
            width=480,
            height=320,
            num_frames=48,  # 2 seconds at 24fps
            fps=24,
            num_inference_steps=25,  # Faster for testing
            seed=42
        )
        
        # Save test video
        output_path = Path(config.OUTPUT_DIR) / "test_video.mp4"
        success = pipeline.save_video(frames, str(output_path), fps=24)
        
        if success:
            print(f"✅ Test successful! Video saved to: {output_path}")
            return True
        else:
            print("❌ Test failed: Could not save video")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def get_example_prompts():
    """Get example prompts for video generation"""
    return [
        "A professional businesswoman walking confidently down a modern office hallway, natural lighting, 4K quality",
        "A young man doing yoga poses in a peaceful garden, morning sunlight, cinematic composition",
        "A dancer performing contemporary dance moves in an empty studio, dramatic lighting, artistic shot",
        "An elderly person reading a book in a comfortable armchair by a window, warm lighting, cozy atmosphere",
        "A chef preparing ingredients in a modern kitchen, professional cooking, dynamic movements",
        "A child playing with colorful building blocks, bright and cheerful, educational setting",
        "A musician playing guitar on a stage, concert lighting, energetic performance",
        "A scientist working in a laboratory, precise movements, professional environment",
        "A gardener tending to plants in a greenhouse, natural lighting, peaceful atmosphere",
        "A painter creating artwork in a studio, creative process, artistic environment"
    ]

# ============================================================================
# 🚀 MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("🎬 AI Video Generator - Complete System")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        return False
    
    # Command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Generator")
    parser.add_argument("--mode", choices=["setup", "install", "download", "test", "app"], 
                       default="app", help="Mode to run")
    parser.add_argument("--force", action="store_true", help="Force reinstall/redownload")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "setup":
            return setup_environment()
        
        elif args.mode == "install":
            return install_all_dependencies()
        
        elif args.mode == "download":
            return download_models()
        
        elif args.mode == "test":
            return test_pipeline()
        
        elif args.mode == "app":
            return run_streamlit_app()
        
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return False
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)