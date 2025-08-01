#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=16gb:ngpus=1

cd $PBS_O_WORKDIR

source qdax050/bin/activate

output_path="outputs/hpc/dcrl_20250728_180401"
echo "model path=$output_path"

