#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=jw1524@ic.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/qdrl-robotics/qdax050/bin/:$PATH     # points to venv python path
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
# source ~/.bashrc
source activate
source /vol/cuda/12.0.0/setup.sh


# Rollout task with different damage situations
# damage definition
damaged_joint_idx=("None" "1" "1 4" "2 5" "3 6")
damaged_joint_action=("None" "1" "1 0" "-0.8 -0.8" "0.6 -0.9")

for i in "${!damaged_joint_idx[@]}"; do
    idx="${damaged_joint_idx[$i]}"
    action="${damaged_joint_action[$i]}"
    echo "Running rollouts with damage_joint_idx=$idx and damaged_joint_action=$action"

    base_command="python -m code.main" ######### add args

    if [ "$idx" == "None" ]; then
            $base_command
        else
            idx_args=$(echo $idx)
            action_args=$(echo $action)
            $base_command --damaged_joint_idx $idx_args --damaged_joint_value $action_args
        fi
done