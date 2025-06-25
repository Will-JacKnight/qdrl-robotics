# Environment
- Branch ```main``` is compatible with ```venv```
- Branch ```qdax050``` is compatible with ```venv050```

# Running jobs on HPCs with apptainer
## If not having apptainer in lab machines
Install apptainer by building its source code in ```/vol/bitbucket/USER_NAME``` directory to avoid hitting the quota limit

By default, the building process creates a ```~/go``` directory, which can easily overrun the quota limit as well. Make sure to export the GOPATH in the ~/.bashrc with the following: 
```
export GOPATH=/vol/bitbucket/USER_NAME/go
```
If ```~/go``` is accidently created and the quota limit is hit, run 
```
chmod -R u+rwX ~/go
```
 to grant write access and safely remove the folder by 
 ```
 rm -rf ~/go
 ```


For more details of container usage with apptainer, please refer to the [repo](https://gitlab.doc.ic.ac.uk/AIRL/airl_tools/container-utils).


# Qdax Bug Fix
1. two minor fix needs to be conducted on qdax lib, where the x= and y= in the original code needs to be removed
```
batch_of_fitnesses = jnp.where(
    batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
)

...

# assign fake position when relevant : num_centroids is out of bound
batch_of_indices = jnp.where(
    addition_condition, batch_of_indices, num_centroids
)
```