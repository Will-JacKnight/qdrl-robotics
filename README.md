# Quality-Diversity Reinforcement Learning for Robotics Damage recovery
This repository is built on top of [QDax library](https://github.com/adaptive-intelligent-robotics/QDax) and [Uncertain Quality Diversity](https://github.com/adaptive-intelligent-robotics/Uncertain_Quality_Diversity).

<img src="docs/images/intact_walking.gif" height="150"/><img src="docs/images/recovery_demo.gif" height="150"/>

The goal for the agent(ant) is to run as fast as possible, even when it's damaged.
- Left: Intact walking behavior
- Right: Recovery behaviour after physical damage to one joint

## Content
### Algorithms
For MAP creation, this project incorporates certain QD algorithms as the following:
- MAPElites
- DCRL-ME (from [MAP-Elites with Descriptor-Conditioned Gradients and Archive Distillation into a Single Policy](https://dl.acm.org/doi/10.1145/3583131.3590503))

For tackling uncertainties in the QD environments, our method uses multi-evaluation sampling where each of the following sampling mechanism is wrapped in a container that can be easily swapped during runtime:
- MAPElites sampling
- Archive Sampling (AS) (from [Uncertain Quality-Diversity: Evaluation methodology and new methods for Quality-Diversity in Uncertain Domains](https://ieeexplore.ieee.org/abstract/document/10120985))
- Extract MAPElites (from [Extract-QD Framework: A Generic Approach forQuality-Diversity in Noisy, Stochastic or Uncertain Domains](https://dl.acm.org/doi/epdf/10.1145/3712256.3726404))

For damage recovery, our method implements [Intelligent Trial and Error](https://www.nature.com/articles/nature14422) using Bayesian Optimisation.

## Environment
### Installation
- always upgrade pip before running pip install:
    ```bash
    pip install --upgrade pip
    ```
- Jax cuda version is set by default for its efficiency in running evaluations in parallel, install the dependencies using:
    ```bash
    pip install -r requirements.txt
    ```
- Though it's advised to run on gpu, adaptation experiments can be run on cpu (very slow) with the following:
    ```bash
    export JAX_PLATFORMS=cpu
    python main.py
    ```

### Configurations
- Default parameters are stored in `./config.json`
- Shell command will override default parameters
- Must specify running algo by setting the flag `--algo_type <op>`, `<op>` currently only supports mapelites or dcrl

### Running the code
Before executing any code, make sure one's under the project directory:
```bash
cd <path-to-repository>
```

For batched jobs, it's advised to run on gpu servers using shell commands, please refer to <hpc_train.sh> and <hpc_adapt.sh> under `scripts/`. All runtime results will be summarised in `outputs/<hpc>` by default.

#### Training the MAP
```bash
qsub scripts/hpc_train.sh  # local runs: ./scripts/hpc_train.sh
```
The mode flag needs to be specified using `--mode training` in `hpc_train.sh`.

#### Damage Adaptation
Model path needs to be explicitly set in `hpc_adapt.sh` for job submission.
```bash
qsub scripts/hpc_adapt.sh # local runs: ./scripts/hpc_adapt.sh
```

#### Metrics Evaluations
The plot results will be presented under `evaluations/`.
```bash
python utils/evaluations.py
```

#### Parameters for Result Evaluations
- Vanilla MAP-Elites: ```--container MAP-Elites_Sampling --dropout-rate 0```
- MAP-Elites Sampling: ```--container MAP-Elites_Sampling --num-samples 10```
- Archive Sampling: ```--container Archive-Sampling  --num-samples 2 --depth 2```
- Extract MAP-Elites: ```--container Extract-MAP-Elites --num-samples 2 --depth 8```

## Notes
- batched_rewards: evaluates all cells in the archive, without any inf/nan values
- repertoire.fitnesses: with empty cells are portrayed as -inf