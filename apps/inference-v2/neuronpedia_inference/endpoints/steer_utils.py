import torch


class OrthogonalProjector:
    """Performs orthogonal projection steering for language model activations.

    This class implements low-rank orthogonal projection-based steering by projecting
    activations onto and orthogonal to a steering direction.

    Attributes:
        steering_vector: The direction to project onto/orthogonal to
        _P: Cached projection matrix
        _orthogonal_complement: Cached orthogonal complement matrix
    """

    def __init__(self, steering_vector: torch.Tensor):
        """Initializes projector with a steering vector.

        Args:
            steering_vector: Vector defining steering direction, shape (d,)
                           where d is activation dimension

        Raises:
            ValueError: If steering vector contains inf/nan values
        """
        self._P = None
        self._orthogonal_complement = None
        self.steering_vector = steering_vector.unsqueeze(1)

    def get_P(self) -> torch.Tensor:
        """Computes or returns cached projection matrix.

        Returns:
            Projection matrix P = vv^T/||v||^2, shape (d,d)

        Raises:
            ValueError: If projection computation fails or results in inf/nan
        """
        if self._P is None:
            self.steering_vector = torch.matmul(
                self.steering_vector, self.steering_vector.T
            )
            self._P = self.steering_vector

            if not torch.isfinite(self._P).all():
                raise ValueError("Projection matrix contains inf or nan values")

        return self._P

    def get_orthogonal_complement(self) -> torch.Tensor:
        """Computes or returns cached orthogonal complement matrix.

        Returns:
            Matrix I-P where P is projection matrix, shape (d,d)

        Raises:
            ValueError: If computation fails
        """
        if self._orthogonal_complement is None:
            P = self.get_P()  # This may raise ValueError
            I = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)  # noqa
            self._orthogonal_complement = I - P
            if not torch.isfinite(self._orthogonal_complement).all():
                raise ValueError(
                    "Orthogonal complement matrix contains inf or nan values"
                )
        return self._orthogonal_complement

    def project(
        self, activations: torch.Tensor, strength_multiplier: float = 1.0
    ) -> torch.Tensor:
        """Projects activations using orthogonal decomposition.

        Decomposes activations into components parallel and orthogonal to steering direction,
        then recombines with optional scaling of parallel component.

        Args:
            activations: Input activations to project, shape (d,)
            strength_multiplier: Scaling factor for parallel component

        Returns:
            Projected activations = (I-P)h + strength*Ph, shape (d,)
        """
        P = self.get_P()
        orthogonal_complement = self.get_orthogonal_complement()
        # use same dtype as activations
        orthogonal_complement = orthogonal_complement.to(activations.dtype)
        P = P.to(activations.dtype)
        return torch.matmul(
            activations, orthogonal_complement.T
        ) + strength_multiplier * torch.matmul(activations, P.T)
