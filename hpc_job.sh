#!/bin/bash
#PBS -lwalltime=10:00:00
#PBS -lselect=1:ncpus=4:mem=64gb:ngpus=1

cd $PBS_O_WORKDIR

source qdax050/bin/activate

# map creation
# python main.py --config config.json --algo_type dcrl --output_path outputs --mode training

# adaptation inspection
python main.py --config config.json --output_path outputs/dcrl_20250704_185243

echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"