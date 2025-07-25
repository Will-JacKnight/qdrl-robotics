
# output_path="outputs/slurm/dcrl_20250710_133450"
output_path="outputs/hpc/dcrl_20250723_160932"
# output_path="outputs/hpc/mapelites_20250724_102129"
# output_path="outputs/slurm/dcrl_20250723_175333"


exp_path="${output_path}/physical_damage"
mkdir -p "$exp_path"

# damage rotation
damaged_joint_idx=("4 5")
damaged_joint_action=("0 0")
damage_desc=("BL_stiff")

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

echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"
