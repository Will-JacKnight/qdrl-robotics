from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.networks.networks import MLP, MLPDC

class ResMLP(MLP):

    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, train: Optional[bool] = False) -> jnp.ndarray:
        hidden = obs
        hidden = nn.Dense(
                    self.layer_sizes[0],
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
        )(hidden)
        if self.dropout_rate > 0.0:
            hidden = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(hidden)

        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                residual = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                residual = self.activation(residual)  # type: ignore
                if self.dropout_rate > 0.0:
                    residual = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(residual)

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=False
                )(hidden) + residual

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
                    
                if self.dropout_rate > 0.0:
                    hidden = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(hidden)
        return hidden

class ResMLPDC(MLPDC):

    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, desc: jnp.ndarray, train: Optional[bool] = False) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, desc], axis=-1)
        hidden = nn.Dense(
                    self.layer_sizes[0],
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
        )(hidden)
        if self.dropout_rate > 0.0:
            hidden = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(hidden)

        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                residual = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                residual = self.activation(residual)  # type: ignore
                if self.dropout_rate > 0.0:
                    residual = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(residual)

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=False
                )(hidden) + residual

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
                    
                if self.dropout_rate > 0.0:
                    hidden = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(hidden)
        return hidden


class DropoutMLP(MLP):
    """
    Extend of QDax MLP module for describing the policy network, dropout activated by default.
    Args:
        - dropout_rate (float): probability of dropout, not keep rate
        - deterministic (bool): set True to disable Dropout, False by default
    """
    dropout_rate: float = 0.2

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
                if self.dropout_rate > 0.0:
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


class DropoutMLPDC(MLPDC):
    """
    Extend of Qdax Descriptor-conditioned MLP module for describing the actor DC network, with dropout activated by default.
    Args:
        - dropout_rate (float): probability of dropout, not keep rate
        - deterministic (bool): set True to disable Dropout, False by default
    """

    dropout_rate: float = 0.2

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
                if self.dropout_rate > 0.0:
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
