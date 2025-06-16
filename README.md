## Qdax Bug Fix
### two minor fix needs to be conducted on qdax lib, where the x= and y= in the original code needs to be removed
```
batch_of_fitnesses = jnp.where(
    batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
)
```

```
# assign fake position when relevant : num_centroids is out of bound
batch_of_indices = jnp.where(
    addition_condition, batch_of_indices, num_centroids
)
```