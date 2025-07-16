# Intelligent Trial and Error
<img src="docs/images/intact_walking.gif" height="150"/><img src="docs/images/recovery_demo.gif" height="150"/>

- Left: Intact walking behavior
- Right: Recovery behaviour after physical damage to a joint

# Environment
## Package Installation
- always upgrade pip before running pip install:
    ```
    pip install --upgrade pip
    ```
- accelerate MAP training (jax==0.4.28 to accelerate using cuda): requirements.txt
- adaptation only (compatible with tinygp): requirements_tgp.txt

- if run adaptation experiments on pure cpu, run the following:
    ```
    export JAX_PLATFORMS=cpu
    python main.py
    ```
## Branches
- Branch ```main``` is compatible with qdax==0.5.0
- Branch ```qdax031``` is compatible with qdax==0.3.1 and ```venv``` environment



# Parameter
## Configurations
- Default parameters are stored in ```./config.json```
- Shell command will override default parameters
- Must specify running algo by setting the flag ```--algo_type <op>```, ```<op>``` currently only supports mapelites or dcrl
- Run mode is set to ```adaptation``` by default. during training, one has to specify the mode flag by ```--mode training``` or use hpc_job.sh instead

## Some Default Parameters
- env_name: 'ant_uni' its reward = forward reward (proportional to forward velocity) + healthy reward - control cost - contact cost
- episode_length: 1000 {maximal rollout length}
- num_iterations: 250, 500, 1000
- grid_shape: (10, 10, 10, 10)
- batch_size: 1024
- damage_joint_idx: [0, 1]    # value between [0,7]
- damage_joint_action: [0, 0.9] # value between [-1,1]
- zero_sensor_idx: null # value between [0,86]
