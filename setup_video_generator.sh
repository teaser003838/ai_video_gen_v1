#!/bin/bash
# AI Video Generator - Quick Setup Script

echo "🎬 AI Video Generator - Quick Setup"
echo "===================================="

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.8 or higher required"
    exit 1
fi

echo "✅ Python version compatible"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements_video_generator.txt

# Check installation
echo "🔍 Verifying installation..."
python3 -c "
import torch
import diffusers
import transformers
print('✅ Core libraries installed')
print(f'PyTorch: {torch.__version__}')
print(f'Diffusers: {diffusers.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo ""
echo "🚀 Setup complete!"
echo ""
echo "Usage:"
echo "  python3 ai_video_generator_complete.py --mode app      # Run web interface"
echo "  python3 ai_video_generator_complete.py --mode test     # Run test"
echo "  python3 ai_video_generator_complete.py --mode setup    # Setup environment"
echo ""
echo "For web interface, run:"
echo "  streamlit run ai_video_generator_complete.py"