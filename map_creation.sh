#!/bin/bash
#PBS -lwalltime=02:00:00
#PBS -lselect=1:ncpus=16:mem=128gb
export PROJECT_ROOT="$HOME/qdrl-robotics"

source "$PROJECT_ROOT/qdax050/bin/activate"

python $PROJECT_ROOT/main.py
# python -c "import jax.numpy as jnp; x = jnp.array([1.0, 2.0, 3.0]); print(x.device)"
echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"