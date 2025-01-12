import torch


@torch.jit.script
def adjugate_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Return adjugate matrix [1].

    Parameters
    ----------
    matrix:
        Input matrix

    Returns
    -------
    torch.Tensor
        Adjugate of input matrix

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Adjugate_matrix
    """

    cofactor = torch.inverse(matrix).T * torch.det(matrix)
    return cofactor.T


def unimodular_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Rescale matrix such that det(ellipses) = 1, in other words, make it unimodular. Doest not work with tensors
    of dtype torch.float64.

    Parameters
    ----------
    matrix:
        Matrix input

    Returns
    -------
    torch.Tensor
        Unimodular version of input matrix.
    """
    val = 1.0 / torch.det(matrix)
    return (torch.sign(val) * torch.pow(torch.abs(val), 1.0 / 3.0))[
        ..., None, None
    ] * matrix
