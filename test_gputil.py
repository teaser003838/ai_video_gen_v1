#!/usr/bin/env python3

"""
Test script to verify GPUtil installation and functionality.
This script tests the GPUtil module and provides alternatives if needed.
"""

import sys
import platform

def test_gputil():
    print("🔍 Testing GPUtil module...")
    
    try:
        import GPUtil
        print("✅ GPUtil module imported successfully")
        
        # Test basic functionality
        try:
            gpus = GPUtil.getGPUs()
            print(f"📊 Found {len(gpus)} GPU(s)")
            
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")
                    print(f"    Memory: {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB")
                    print(f"    Temperature: {gpu.temperature}°C")
                    print(f"    Load: {gpu.load*100:.1f}%")
            else:
                print("  No GPUs detected")
                
        except Exception as e:
            print(f"⚠️ Error accessing GPU info: {str(e)}")
            
        return True
        
    except ImportError as e:
        print(f"❌ GPUtil module not found: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Error testing GPUtil: {str(e)}")
        return False

def test_alternatives():
    print("\n🔧 Testing alternative GPU detection methods...")
    
    # Test torch GPU detection
    try:
        import torch
        print("✅ PyTorch imported successfully")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️ CUDA not available")
            
    except ImportError:
        print("❌ PyTorch not available")
    except Exception as e:
        print(f"❌ Error testing PyTorch: {str(e)}")
        
    # Test basic system info
    try:
        import psutil
        print(f"✅ System info available")
        memory = psutil.virtual_memory()
        print(f"   RAM: {memory.total / 1024**3:.1f}GB total, {memory.available / 1024**3:.1f}GB available")
        
    except ImportError:
        print("❌ psutil not available")
    except Exception as e:
        print(f"❌ Error testing system info: {str(e)}")

def provide_alternative_code():
    """Provide alternative code for GPUtil functionality"""
    print("\n💡 Alternative code for GPUtil functionality:")
    print("=" * 50)
    
    alternative_code = '''
# Alternative to GPUtil for GPU detection
import sys
import platform

try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        gpu_name = 'None'
        gpu_memory_gb = 0
except ImportError:
    gpu_available = False
    gpu_name = 'None'
    gpu_memory_gb = 0

# Alternative system info
try:
    import psutil
    memory = psutil.virtual_memory()
    system_info = {
        'platform': platform.system(),
        'python_version': sys.version,
        'total_ram_gb': round(memory.total / (1024**3), 2),
        'available_ram_gb': round(memory.available / (1024**3), 2),
        'cuda_available': gpu_available,
        'gpu_name': gpu_name,
        'gpu_memory_gb': round(gpu_memory_gb, 2)
    }
except ImportError:
    system_info = {
        'platform': platform.system(),
        'python_version': sys.version,
        'cuda_available': False,
        'gpu_name': 'None',
        'gpu_memory_gb': 0
    }

print("System Information:")
for key, value in system_info.items():
    print(f"  {key}: {value}")
'''
    
    print(alternative_code)

if __name__ == "__main__":
    print("🧪 GPU Detection Test")
    print("=" * 40)
    
    # Test GPUtil
    gputil_works = test_gputil()
    
    # Test alternatives
    test_alternatives()
    
    # Provide alternative code if GPUtil doesn't work
    if not gputil_works:
        provide_alternative_code()
    
    print("\n✅ Test completed!")