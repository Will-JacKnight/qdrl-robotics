#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=16gb:ngpus=1:gpu_type=L40S

cd $PBS_O_WORKDIR

source qdax050/bin/activate

python main.py --output_path outputs/final/dcrl_20250904_232254

python main.py --output_path outputs/final/dcrl_20250904_232353

python main.py --output_path outputs/final/dcrl_20250904_232631

python main.py --output_path outputs/final/dcrl_20250904_234028
