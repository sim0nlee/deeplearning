from dks.examples.haiku.modified_resnet import ModifiedResNet
import haiku as hk
import jax
import jax.numpy as jnp

def func(batch, is_training):
    model = ModifiedResNet(
        num_classes=1000,
        depth=152,
        activation_name="leaky_relu",
        shortcut_weight=0.9,
    )
    return model(batch, is_training=is_training)

forward = hk.without_apply_rng(hk.transform_with_state(func))

rng = jax.random.PRNGKey(42)

image = jnp.ones([2, 224, 224, 3])
params, state = forward.init(rng, image, is_training=True)
logits, state = forward.apply(params, state, image, is_training=True)