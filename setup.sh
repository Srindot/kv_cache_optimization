#!/bin/bash

# KV Cache Optimization Framework Setup Script
echo "Setting up KV Cache Optimization Framework..."
echo "============================================"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda detected"
    
    echo ""
    echo "Creating conda environment..."
    conda env create -f environment.yml
    echo ""
    echo "✓ Environment created successfully!"
    echo "To activate: conda activate kv_cache_optimization"
else
    echo "⚠ Conda not found. Installing dependencies with pip..."
    pip install -r requirements.txt
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p results/plots
mkdir -p example_results

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate your conda environment (if created)"
echo "2. Run: python run_experiments.py --help"
echo "3. Or try: python example.py"
echo "4. For analysis: jupyter notebook analysis_notebook.ipynb"
echo ""
echo "For more information, see README.md"