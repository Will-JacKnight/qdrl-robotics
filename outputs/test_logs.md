# Test Logs

## Converged Grids
- mapelites_20250701_152736

## Reward tests

The tests carried before the following is trained on full brax reward (including penaltization)
- dcrl_20250703_114735
- mapelites_20250701_152736

The following tests are trained on forward reward only:
- dcrl_20250704_185243
- dcrl_20250710_134938
- dcrl_20250710_133450 (slurm, 3000 iterations)
- dcrl_20250711_113903 (slurm, 5000 iterations)

Earlier dcrls have implementation issues with actor_dc_network which may result them to degrade to DCG-ME, and is fixed for later models.


Correct implementation and ones trained on forward reward:
- hpc/dcrl_20250723_160932 (no dropouts)
- hpc/mapelites_20250724_102129 (dropouts)
- slurm/dcrl_20250723_175333 (dropouts)
