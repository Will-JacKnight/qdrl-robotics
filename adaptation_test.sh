#!/bin/bash
#PBS -lwalltime=02:00:00
#PBS -lselect=1:ncpus=4:mem=64gb:ngpus=1


# output_path="outputs/slurm/dcrl_20250710_133450"
output_path="outputs/hpc/dcrl_20250723_160932"
python main.py --config config.json --output_path $output_path  --exp_path $output_path --damage_type physical
