
# output_path="outputs/slurm/dcrl_20250710_133450"
# output_path="outputs/hpc/dcrl_20250723_160932"
output_path="outputs/slurm/dcrl_20250723_175333"
python main.py --config config.json --output_path $output_path  --exp_path $output_path --damage_type physical
