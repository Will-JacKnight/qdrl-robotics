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

class ForwardStepRewardWrapper(Wrapper):
    """
    Wrapper that only include forward step reward.

    - *forward step reward*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
    time between actions - the default *dt = 0.05*. This reward would be
    positive if the ant moves forward (right) desired.
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


class JourneyAwayRewardWrapper(Wrapper):
    """
    Wrapper that only include journey away reward.
    - *journey_away_reward*: A reward of moving forward which is measured as
    the Euclidean Distance between current location and origin location.
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
        x_t = state.metrics["x_position"]
        y_t = state.metrics["y_position"]

        new_reward = jnp.sqrt(x_t ** 2 + y_t ** 2)
        return state.replace(reward=new_reward)