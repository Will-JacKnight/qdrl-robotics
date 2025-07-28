#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=16gb:ngpus=1

cd $PBS_O_WORKDIR

source qdax050/bin/activate

# map training
# python main.py --config config.json --algo_type dcrl --output_path outputs/hpc --mode training

# # adaptation inspection
# output_path="outputs/hpc/mapelites_20250727_211830"
output_path="outputs/hpc/dcrl_20250727_210952"
echo "Damage_joint_idx=$output_path"
# python main.py --config config.json --output_path $output_path  --exp_path $output_path --damage_type physical

exp_path="${output_path}/physical_damage"
rm -rf "$exp_path"
mkdir -p "$exp_path"

# damage rotation
damaged_joint_idx=("0 1" "4 5" "4 5 6 7" "0 1 6 7")
damaged_joint_action=("0 0" "0 0" "0 0 0 0" "0 0 0 0")
damage_desc=("FL_loose" "BL_loose" "BL_BR_loose" "FL_BR_loose")

for i in "${!damaged_joint_idx[@]}"; do
    echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    echo "Experiment ${damage_desc[$i]}:"

    idx="${damaged_joint_idx[$i]}"
    action="${damaged_joint_action[$i]}"
    echo "Damage_joint_idx=$idx"
    echo "Damaged_joint_action=$action"

    damage_path="${exp_path}/${damage_desc[$i]}"
    mkdir -p "$damage_path"
    python main.py --config config.json --output_path $output_path --exp_path $damage_path --damage_joint_idx $idx --damage_joint_action $action --damage_type physical
    
done


exp_path="${output_path}/sensory_damage"
rm -rf "$exp_path"
mkdir -p "$exp_path"
python main.py --config config.json --output_path $output_path --exp_path $exp_path --damage_type sensory

echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"