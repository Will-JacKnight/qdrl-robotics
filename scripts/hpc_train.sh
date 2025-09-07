#!/bin/bash
#PBS -lwalltime=07:00:00
#PBS -lselect=1:ncpus=1:mem=64gb:ngpus=1:gpu_type=L40S

# PBS_O_WORKDIR is where the job's submitted
cd $PBS_O_WORKDIR

source qdax050/bin/activate

timestamp=$(date +"%Y%m%d_%H%M%S")
output_path="outputs/final/dcrl_$timestamp"
echo "Output model to path: $output_path"

seed=$((RANDOM % 1001))

## vanilla dcrl-ME 
# python main.py --algo_type dcrl --output_path $output_path --mode training \
#     --container MAP-Elites-Sampling --dropout-rate 0 \
#     --seed $seed

## dropout dcrl-ME
# python main.py --algo_type dcrl --output_path $output_path --mode training \
#     --container MAP-Elites-Sampling \
#     --seed $seed

## mapelite-sampling 
# python main.py --algo_type dcrl --output_path $output_path --mode training \
#     --container MAP-Elites-Sampling --num-samples 10 \
#     --seed $seed


## extract-mapelites
python main.py --algo_type dcrl --output_path $output_path --mode training \
    --container Extract-MAP-Elites --num-samples 2 --depth 8 \
    --seed $seed


# Check if the training failed and exit if so
if [ $? -ne 0 ]; then
    echo "Training failed with an internal error. Exiting script."
    exit 1
fi

echo "%%%%%%%%%%%%%%%Training Complete%%%%%%%%%%%%%%%"

# physical damage configs
damaged_joint_idx=("0 1" "4 5" "4 5 6 7" "0 1 6 7")
damaged_joint_action=("0 0" "0 0" "0 0 0 0" "0 0 0 0")
damage_desc=("FL_loose" "BL_loose" "BL_BR_loose" "FL_BR_loose")


exp_path="${output_path}/physical_damage"
mkdir -p "$exp_path"

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
        --damage_joint_idx $idx --damage_joint_action $action --damage_type physical \
        --num-reevals 16 --reeval-scan-size 8
    
    if [ $? -ne 0 ]; then
    echo "Physical damage adaptation failed with an internal error. Exiting script."
    exit 1
    fi
    
done

# sensory damage configs
zero_sensor_idx=("5 6 19 20" "9 10 23 24" "8 12 19 25" "3 11 17 20")
damage_desc=("FL" "BL" "Rand1" "Rand2")

exp_path="${output_path}/sensory_damage"
mkdir -p "$exp_path"

for i in "${!zero_sensor_idx[@]}"; do
    echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    echo "Experiment ${damage_desc[$i]}:"

    idx="${zero_sensor_idx[$i]}"
    echo "Damage_joint_idx=$idx"

    damage_path="${exp_path}/${damage_desc[$i]}"
    mkdir -p "$damage_path"
    python main.py --output_path $output_path --exp_path $damage_path \
        --zero_sensor_idx $idx --damage_type sensory \
        --num-reevals 16 --reeval-scan-size 8 
        
    if [ $? -ne 0 ]; then
        echo "Sensory damage adaptation failed with an internal error. Exiting script."
        exit 1
    fi
done

echo "%%%%%%%%%%%%%%%Adaptation Complete%%%%%%%%%%%%%%%"