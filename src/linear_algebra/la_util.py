from jax import np as jnp


def textbook_linear_regression(A, b):
    # https://www.cs.sfu.ca/~mark/ftp/Tip2011/tip2011.pdf (section 9)
    # 1) Ax = b; solve for x
    # 2) A.T•A•x = A.T•b
    # 3) x = (A.T•A)^-1 • (A.T•b)
    x = pinv(A.T.dot(A)).dot(A.T.dot(b))
    return x


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
