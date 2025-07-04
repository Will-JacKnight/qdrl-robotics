#!/bin/bash
#PBS -lwalltime=10:00:00
#PBS -lselect=1:ncpus=4:mem=64gb:ngpus=1

cd $PBS_O_WORKDIR

source qdax050/bin/activate

python main.py --config config.json --algo_type dcrl --output_path outputs --mode training



# export PROJECT_ROOT="$HOME/qdrl-robotics"

# cd $PROJECT_ROOT
# source $PROJECT_ROOT/qdax050/bin/activate

# # python -c "import jax.numpy as jnp; x = jnp.array([1.0, 2.0, 3.0]); print(x.device)"
# python main.py --config "$PROJECT_ROOT/config.json" --algo_type mapelites --output_path "$PROJECT_ROOT/outputs"

echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"