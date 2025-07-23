from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.networks.networks import MLP, MLPDC

class CustomMLP(MLP):
    """
    Extend of QDax MLP module, with dropout activated by default.
    Args:
        - dropout_rate (float): probability of dropout, not keep rate
        - deterministic (bool): set True to disable Dropout, False by default
    """
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, train: Optional[bool] = False) -> jnp.ndarray:
        hidden = obs
        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)  # type: ignore
                hidden = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(hidden)

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden


class CustomMLPDC(MLPDC):
    """
    Extend of Qdax Descriptor-conditioned MLP module, with dropout activated by default.
    Args:
        - dropout_rate (float): probability of dropout, not keep rate
        - deterministic (bool): set True to disable Dropout, False by default
    """

    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, desc: jnp.ndarray, train: Optional[bool] = False) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, desc], axis=-1)
        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)  # type: ignore
                hidden = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(hidden)

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden
