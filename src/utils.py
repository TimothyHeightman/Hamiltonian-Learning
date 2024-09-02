"""
Common utility functions
"""

import functools
import numpy as np


def kron(args):
    """
    Kronecker product with variable-length arguments
    """
    backend = np if isinstance(args[0], np.ndarray) else torch
    return functools.reduce(backend.kron, args)


def integer_to_binary(arr, length):
    """
    Convert an integer array to a binary array.

    Args:
        arr (array-like): array of integers
        length (int): length of the binary bit-strings

    Returns:
        array-like: binary representation of the integer array,
            with shape (s, length) where s is the shape of the input
    """
    return (((arr[..., None] & (1 << np.arange(length))[::-1])) > 0).astype(np.float32)

