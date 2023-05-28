import numpy as np


"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None # (4880,)
        self.components = None # (4880, 40)

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        self.mean = np.mean(X, axis=0)
        X_shifted = X - self.mean
        cov = X_shifted.T @ X_shifted # X^T @ X, Note that scalar does not matter
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        X_shifted = X - self.mean
        return np.dot(X_shifted, self.components)

    def reconstruct(self, X):
        #TODO: 2%
        X_transformed = self.transform(X)
        return np.dot(X_transformed, self.components.T) + self.mean
