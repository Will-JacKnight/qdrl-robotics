import jax

from qdax import environments


def run_dcrl(env_name,  #
             episode_length, #
             policy_hidden_layer_sizes, #
             batch_size, #
             num_iterations, #
             grid_shape, ##
             min_descriptor, #
             max_descriptor, #
             iso_sigma, #
             line_sigma, #
             ga_batch_size,
             dcrl_batch_size, 
             ai_batch_size, 
             log_period, 
             key):
    
    # Init environment
    env = environments.create(env_name, episode_length=episode_length)
    offset=environments.reward_offset[env_name]

    reset_fn = jax.jit(env.reset)

    
    # return repertoire, metrics, env, policy_network