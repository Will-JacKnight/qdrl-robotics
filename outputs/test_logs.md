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


## Dropout rate comparison:
- hpc/dcrl_20250723_160932 (no dropouts)
- slurm/dcrl_20250723_175333 (dropout_rate: 0.1)
- hpc/dcrl_20250727_210952 (dropout_rate: 0.2)

- hpc/mapelites_20250724_102129 (dropout_rate: 0.1)
- hpc/mapelites_20250727_211830 (no dropouts)

## random damage injection in MAP training:
- hpc/dcrl_20250728_180401 (physical damage only, with training_damage_rate:0.1, dropout_rate: 0.2)
- hpc/dcrl_20250731_153529 (random damage injection (including resets, noise, intensity) at training_damage_rate=0.05: dropout_rate: 0.2)
- (curriculum based learning: increasing training_damage_rate from 0.05 to 0.85, seldom 100% failure episodes)

## ResNet + residuals Dropout Policies
 - hpc/dcrl_20250811_152438 (ai-emitter projection layer + 2 Res layers + action projection layer, dropouts applied on every layer)
 - hpc/dcrl_20250811_174154 (ai-emitter projection layer + layerNorm layer + 1 Res layer + action projection layer, dropouts applied on every layer)

 ## Multi-evaluations to solve ME training stochasticity
- hpc/dcrl_20250723_160932 (no dropouts)
- hpc/dcrl_20250727_210952 (single eval, dropouts except for the output layer)
- hpc/dcrl_20250813_213310 (single eval, dropouts on every layer)
- hpc/dcrl_20250813_212529 (10 parallel evals and averagging on reward, dropouts on every layer)
- hpc/dcrl_20250814_111147 (1/10 env_steps, 10 parallel evals and averagging on reward, dropouts on every layer)
 