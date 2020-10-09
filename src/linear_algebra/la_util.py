from jax import np as jnp


def vector_magnitudes(vecs):
    """ vectorized, differentiable function for vector
    magnitudes
    """
    return jnp.sqrt(jnp.einsum("...i,...i", vecs, vecs))


def normalize_vectors(vecs):
    """ differentiable function for vector
    normalizing vectors
    """
    mags = vector_magnitudes(vecs)
    normals = vecs / (mags[:, jnp.newaxis])
    return normals


def normalize_vector(vec):
    return vec / jnp.linalg.norm(vec)
