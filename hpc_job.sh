#!/bin/bash
#PBS -lwalltime=10:00:00
#PBS -lselect=1:ncpus=4:mem=64gb:ngpus=1

cd $PBS_O_WORKDIR

source qdax050/bin/activate

# map creation
# python main.py --config config.json --algo_type dcrl --output_path outputs --mode training

# adaptation inspection
output_path="outputs/dcrl_20250704_185243"
python main.py --config config.json --output_path $output_path

# # damage rotation
# output_path="outputs/dcrl_20250704_185243"

# damaged_joint_idx=("0 1" "0 1" "4 5 6 7" "0 1 6 7")
# damaged_joint_action=("0 0.9" "-0.8 0.9" "0 0.9 0 0.9" "0 0.9 0 0.9")

# for i in "${!damaged_joint_idx[@]}"; do
#     echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

#     idx="${damaged_joint_idx[$i]}"
#     action="${damaged_joint_action[$i]}"
#     echo "Damage_joint_idx=$idx"
#     echo "Damaged_joint_action=$action"

#     exp_path="${output_path}/exp$((i+1))"
#     mkdir -p "$exp_path"
#     python main.py --config config.json --output_path $exp_path
    
# done

echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"