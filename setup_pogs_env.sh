#!/bin/bash

# Define the absolute path to your conda environment
export POGS_ENV_ROOT="/home/pi0/miniconda3/envs/pogs_env"

# 1. Add the environment's bin to PATH (fixes 'ninja not found' and 'ns-* command not found')
export PATH="$POGS_ENV_ROOT/bin:$PATH"

# 2. Set Library Paths (fixes 'cannot find -lcudart' linker errors during training)
export LIBRARY_PATH="$POGS_ENV_ROOT/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$POGS_ENV_ROOT/lib:$LD_LIBRARY_PATH"

echo "✅ POGS Environment Configured"
echo "   - PATH updated"
echo "   - LIBRARY_PATH set for compilation"
echo "   - LD_LIBRARY_PATH set for runtime"
