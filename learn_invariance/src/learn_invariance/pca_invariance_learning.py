import numpy as np
from sklearn.decomposition import PCA


def project_to_plane(plane_components, params):
    """
    https://math.stackexchange.com/questions/185546/how-to-project-a-n-dimensional-point-onto-a-2-d-subspace
    with the simplification that M^T @ M = I
    https://math.stackexchange.com/questions/1121812/orthogonal-rectangular-matrix

    Args:
        plane_components: A matrix [n, m] where n >= m, whether each column vector is orthonormal
        params: [n_samples, n]

    Returns:

    """
    return np.dot(params, plane_components) @ plane_components.T


def sample_plane(plane_components, low, high, n_samples: int, rng: np.random.RandomState):
    """

    Args:
        plane_components: A matrix NxM where N >= M, whether each column vector is orthonormal
        n_samples:

    Returns:

    """
    params = rng.uniform(low, high, size=[n_samples, low.size])  # [n_samples, n]
    projected_params = project_to_plane(plane_components, params)
    return projected_params


def inv_loss_plane(plane_components, transform_test):
    projected_params = project_to_plane(plane_components, transform_test)
    inv_loss = transform_test - projected_params
    return inv_loss


def main():
    np.set_printoptions(suppress=True, precision=4, linewidth=250)
    trans = np.random.randn(10, 3)
    rot2 = np.random.randn(10, 2) * 1e-3
    rot = np.random.randn(10, 1)

    params = np.concatenate([trans, rot2, rot], -1)
    params_normalized = params / np.linalg.norm(params, axis=-1, keepdims=True)

    pca = PCA()

    pca.fit(params_normalized)

    valid_components = pca.explained_variance_ > 1e-6

    plane_components = pca.components_[valid_components]
    plane_components = plane_components.T
    plane_components, _ = np.linalg.qr(plane_components)
    print(f"{plane_components=}")

    # Show how to sample
    rng = np.random.RandomState(0)
    lim = np.array([1, 1, 1, np.pi, np.pi, np.pi])
    samples = sample_plane(plane_components, -lim, lim, n_samples=5, rng=rng)
    print(f"{samples=}")

    transform_test = np.array([0.1, 0.2, 0.3, 0.4, 0, 0])
    loss = inv_loss_plane(plane_components, transform_test)
    print(f"{loss=}")


if __name__ == '__main__':
    main()
