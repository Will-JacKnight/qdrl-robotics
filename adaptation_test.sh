
output_path="outputs/hpc/dcrl_20250723_160932"

exp_path="${output_path}/physical_damage"
mkdir -p "$exp_path"

# damage rotation
damaged_joint_idx=("4 5")
damaged_joint_action=("0 0")
damage_desc=("BL_loose")

echo "Experiment ${damage_desc}:"

idx="${damaged_joint_idx}"
action="${damaged_joint_action}"
echo "Damage_joint_idx=$idx"
echo "Damaged_joint_action=$action"

damage_path="${exp_path}/${damage_desc}"
mkdir -p "$damage_path"
python main.py --config config.json --output_path $output_path --exp_path $damage_path --damage_joint_idx $idx --damage_joint_action $action --damage_type physical
    

echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"
