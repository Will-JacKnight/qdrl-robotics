# Intelligent Trial and Error
<img src="docs/images/intact_walking.gif" height="150"/><img src="docs/images/recovery_demo.gif" height="150"/>

The goal for the agent is to run as fast as possible, even when it's damaged.
- Left: Intact walking behavior
- Right: Recovery behaviour after physical damage to a joint

# Environment
## Package Installation
- always upgrade pip before running pip install:
    ```
    pip install --upgrade pip
    ```
- Jax cuda version is set by default for its efficiency in running evaluations in parallel, install the dependencies using:
    ```
    pip install -r requirements.txt
    ```
- Though it's advised to run on gpu, adaptation experiments can be run on cpu (very slow) with the following:
    ```
    export JAX_PLATFORMS=cpu
    python main.py
    ```
- For batched jobs, it's advised to run on gpu servers using shell commands, please refer to <hpc_jobs.sh> and <slurm_job.sh>.

## Branches
- Branch ```main``` is compatible with qdax==0.5.0 (up to date)
- Branch ```qdax031``` is compatible with qdax==0.3.1 (obsolete)



# Parameters
## Configurations
- Default parameters are stored in ```./config.json```
- Shell command will override default parameters
- Must specify running algo by setting the flag ```--algo_type <op>```, ```<op>``` currently only supports mapelites or dcrl
- Run mode is set to ```adaptation``` by default. during training, one has to specify the mode flag by ```--mode training``` or use hpc_job.sh instead

## Some Default Parameters
- episode_length: 1000 {maximal rollout length}
- grid_shape: (10, 10, 10, 10)
- damage_joint_idx: [0, 1]    # value between [0,7]
- damage_joint_action: [0, 0.9] # value between [-1,1]
- zero_sensor_idx: null # value between [0,86]

## Notes
- batched_rewards: evaluates all cells in the archive, without any inf/nan values
- repertoire.fitnesses: with empty cells are portrayed as -inf