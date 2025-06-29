# Environment
- Modify the requirements.txt to install dependencies of different versions of qdax.
- Branch ```main``` is compatible with qdax==0.3.1
- Branch ```qdax050``` is compatible with ```venv050```

# Parameter 
## Configurations
- Default parameters are stored in ```./config.json```
- Shell command will override default parameters
- Must specify running algo by setting the flag ```--algo_type <op>```, ```<op>``` can be either mapelites or dcrl

## Some Default Parameters
- env_name: 'ant_uni' its reward = forward reward (proportional to forward velocity) + healthy reward - control cost - contact cost
- episode_length: 1000 {maximal rollout length}
- num_iterations: 250, 500, 1000
- grid_shape: (10, 10, 10, 10)
- batch_size: 1024
- damage_joint_idx: [0, 1]    # value between [0,7]
- damage_joint_action: [0, 0.9] # value between [-1,1]