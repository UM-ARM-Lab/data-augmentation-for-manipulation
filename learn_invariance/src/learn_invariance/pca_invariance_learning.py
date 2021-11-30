import numpy as np
from sklearn.decomposition import PCA

np.set_printoptions(suppress=True, precision=4, linewidth=250)

trans = np.random.randn(10,3)
rot2 = np.random.randn(10,2)*1e-3
rot = np.random.randn(10,1)

params = np.concatenate([trans, rot2, rot], -1)
params_normalized = params /np.linalg.norm(params, axis=-1, keepdims=True)

pca = PCA()

pca.fit(params_normalized)

valid_components = pca.explained_variance_ > 1e-6
invalid_components = np.logical_not(valid_components)

top_components = pca.components_[valid_components]
top_components = top_components.T
print(top_components)

# Show how to evaluate a test point, and get a gradient
b = np.array([0.1, 0.2, 0.3, 0.4, 0, 0])

x = np.linalg.lstsq(top_components, b, rcond=None)[0]

gradient = b - top_components@x

print(f"{gradient=}")
