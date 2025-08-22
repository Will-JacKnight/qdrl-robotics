#!/bin/bash
#PBS -lwalltime=10:00:00
#PBS -lselect=1:ncpus=2:mem=32gb:ngpus=1:gpu_type=L40S

# PBS_O_WORKDIR is where the job's submitted
cd $PBS_O_WORKDIR

source ../qdax050/bin/activate

# map training
## mapelite-sampling 

# python main.py --config config.json --algo_type dcrl --output_path outputs/hpc --mode training \
#     --container-name mapelites_sampling

## mapelite-sampling 

python main.py --config config.json --algo_type dcrl --output_path outputs/hpc --mode training \
    --container-name archive_sampling --num-samples 2 --depth 2


## mapelite-sampling 
# python main.py --config config.json --algo_type dcrl --output_path outputs/hpc --mode training \
#     --container-name extract_mapelites --num-samples 2 --depth 8

echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"