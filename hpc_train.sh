#!/bin/bash
#PBS -lwalltime=10:00:00
#PBS -lselect=1:ncpus=2:mem=32gb:ngpus=1

cd $PBS_O_WORKDIR

source qdax050/bin/activate

# map training
python main.py --config config.json --algo_type dcrl --output_path outputs/hpc --mode training

echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"