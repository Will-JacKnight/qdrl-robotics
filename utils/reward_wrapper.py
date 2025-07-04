from typing import Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper

from qdax.tasks.brax.v1.wrappers.base_wrappers import QDWrapper

# name of the forward/velocity reward
FORWARD_REWARD_NAMES = {
    "ant_uni": "reward_forward",
    "halfcheetah": "reward_run",
    "walker2d": "reward_forward",
    "hopper": "reward_forward",
    "humanoid": "reward_linvel",
}

class WalkRewardOnlyWrapper(Wrapper):
    """
    wrapper that only include forward reward
    """

    def __init__(self, env: Env, env_name: str):
        super().__init__(env)
        self._env_name = env_name
        self._forward_reward_name = FORWARD_REWARD_NAMES[env_name]
    
    @property
    def name(self) -> str:
        return self._env_name
    
    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        new_reward = state.metrics[self._forward_reward_name]
        return state.replace(reward=new_reward)