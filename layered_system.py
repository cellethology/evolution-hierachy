import numpy as np
from scipy.special import expit


def uniform_sphere_gaussian(n_samples, dim):
    # Generate Gaussian samples
    X = np.random.randn(n_samples, dim)

    # Normalize and scale
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


class LayeredSystem:
    def __init__(
        self,
        dim=20,
        max_depth=30,
        rank_fraction=1.0,
        use_sigmoid=False,
        seed=None,
    ):
        """
        Initialize a layered system.
        """
        if seed is not None:
            np.random.seed(seed)

        self.dim = dim
        self.max_depth = max_depth
        self.rank_fraction = rank_fraction
        self.use_sigmoid = use_sigmoid

        # Initialize matrices
        self.matrices = [
            self._generate_matrix_with_rank(dim, rank_fraction)
            for _ in range(max_depth)
        ]

    def _generate_matrix_with_rank(self, input_dim, rank_fraction=1.0):
        """Generate a matrix with controlled rank"""
        if rank_fraction == 1.0:
            s = np.random.randn(input_dim, input_dim)
            return s
        else:
            target_rank = int(input_dim * rank_fraction)
            U = np.random.randn(input_dim, input_dim)
            V = np.random.randn(input_dim, input_dim)
            U, _ = np.linalg.qr(U)
            V, _ = np.linalg.qr(V)
            s = np.zeros(input_dim)
            s[:target_rank] = 1.0
            return U @ np.diag(s) @ V.T

    def _forward_pass(self, X, start_depth=0, end_depth=None):
        """Propagate vectors through layers"""
        if end_depth is None:
            end_depth = self.max_depth

        outputs = [X]
        current = X.copy()

        for d in range(start_depth, end_depth):
            current = current @ self.matrices[d].T
            if self.use_sigmoid:
                current = expit(current)
            outputs.append(current.copy())

        return outputs

    def _backward_pass(self, X_final, start_depth=0, end_depth=None):
        """
        Compute the best approximation between start and end for X_final
        """
        if end_depth is None:
            end_depth = self.max_depth

        outputs = [X_final]
        current = X_final.copy()

        for d in reversed(range(start_depth, end_depth)):
            if self.use_sigmoid:
                eps = 1e-15
                current = np.clip(current, eps, 1.0 - eps)
                current = LayeredSystem.logit(current)

            # Invert the matrix multiplication: forward was (current @ M[d].T),
            # so backward is multiply by pinv(M[d].T).
            # pinv(M[d].T) = (pinv(M[d]))^T, but we can just do pinv on the transpose directly.
            M_d_T_pinv = np.linalg.pinv(self.matrices[d].T)
            current = current @ M_d_T_pinv
            outputs.append(current.copy())

        # Reverse the outputs to get the correct order
        outputs.reverse()
        return outputs

    @staticmethod
    def logit(u):
        """Inverse of sigmoid: logit(u) = ln(u/(1-u))."""
        return np.log(u / (1.0 - u))

    def _compute_cossim(self, vectors, reference):
        """Compute angles between vectors and a reference vector"""
        cossim = np.sum(vectors * reference, axis=1) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(reference)
        )
        return cossim
