#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=jw1524@ic.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/qdrl-robotics/qdax050/bin/:$PATH     # points to venv python path
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
# source ~/.bashrc
source activate
source /vol/cuda/12.9.0/setup.sh
# . /vol/cuda/12.0.0/setup.sh

# map training
# python main.py --config config.json --algo_type dcrl --output_path outputs/slurm --mode training

# adaptation inspection
output_path="outputs/slurm/dcrl_20250723_175333"
# # python main.py --config config.json --output_path $output_path  --exp_path $output_path --damage_type physical

exp_path="${output_path}/physical_damage"
rm -rf "$exp_path"
mkdir -p "$exp_path"

# damage rotation
damaged_joint_idx=("0 1" "0 1" "4 5 6 7" "0 1 6 7")
damaged_joint_action=("0 0.9" "-0.8 0.9" "0 0.9 0 0.9" "0 0.9 0 0.9")
damage_desc=("FL_stiff" "FL_stiff_2" "BL_BR_stiff" "FL_BR_stiff")

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

# interactive job
# srun --pty --gres=gpu:1 bash

echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"