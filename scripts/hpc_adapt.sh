#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=16gb:ngpus=1

cd $PBS_O_WORKDIR

source qdax050/bin/activate

# adaptation inspection
output_path="outputs/hpc/dcrl_20250827_170158"

echo "model path=$output_path"

###############

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
    python main.py --output_path $output_path --exp_path $damage_path \
        --damage_joint_idx $idx --damage_joint_action $action --damage_type physical
    
done

###############

exp_path="${output_path}/sensory_damage"
rm -rf "$exp_path"
mkdir -p "$exp_path"

zero_sensor_idx=("5 6 19 20" "9 10 23 24" "8 12 19 25" "3 11 17 20")
damage_desc=("FL" "BL" "Rand1" "Rand2")

for i in "${!zero_sensor_idx[@]}"; do
    echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    echo "Experiment ${damage_desc[$i]}:"

    idx="${zero_sensor_idx[$i]}"
    echo "Damage_joint_idx=$idx"

    damage_path="${exp_path}/${damage_desc[$i]}"
    mkdir -p "$damage_path"
    python main.py --output_path $output_path --exp_path $damage_path \
        --zero_sensor_idx $idx --damage_type sensory
    
done

echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"