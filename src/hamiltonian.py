"""
This file contains the base class for building Hamiltonian models to be learned.
"""

import abc
from typing import List
import functools

import numpy as np
import scipy
import jax
import jax.numpy as jnp

from .utils import kron


def set_at(a, index, val):
    if isinstance(a, jax.Array):
        term = jnp.zeros_like(a)
        term = term.at[index].set(1)
    else:
        term = np.zeros_like(a)
        term[index] = 1

    return a + val * term


class Hamiltonian(abc.ABC):
    """
    Abstract class for different Hamiltonians
    """

    def __init__(self, n: int, ops: List[np.ndarray], coeffs: List[np.ndarray] = None):
        """Contruct attributes of model.

        Args:
            n: number of qudits
            ops: the operators in the Hamiltonian; each element of
                the list must have shape (k, n), where n is the number of qudits
                and k is the number of operators. Operators in the same
                array have the same parameters, but operators in different arrays
                have different parameters. Each entry in the array is an index into
                the operator basis of the Hamiltonian's Hilbert space
            coeffs: the (relative) coefficients of the operators in the Hamiltonian
                in order to impose constraints on the parameter space; each element of the 
                list must have shape (k,). If not provided, 1 is assumed as the coefficient, 
                implying no constraints
        """
        self.n = n
        self.ops = ops
        
        if coeffs is not None:
            self.coeffs = coeffs
        else:
            self.coeffs = [np.ones(op.shape[0]) for op in ops]

    @property
    def num_parameters(self) -> int:
        """
        Get the number of parameters in the Hamiltonian

        Returns:
            int: number of parameters in the Hamiltonian
        """
        return len(self.ops)

    @property
    def num_observables(self) -> int:
        """
        Get the total number of Pauli observables in the Hamiltonian

        Returns:
            int: number of Pauli observables in the Hamiltonian
        """
        return sum(op.shape[0] for op in self.ops)

    @property
    @abc.abstractmethod
    def d(self) -> int:
        """
        Get the dimension of the Hilbert space of each qudit
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def operator_basis(self):
        """
        Get the operator basis of the Hilbert space of each qudit
        """
        raise NotImplementedError

    def build_dense_hamiltonian(self, params: list) -> np.ndarray:
        """
        Get the dense matrix representation of the Hamiltonian

        Args:
            params (list[float]): the coefficients of the Hamiltonian;
                the list must have length self.num_parameters

        Returns:
            np.ndarray: the matrix representation of the Hamiltonian
        """
        H = None

        for param, op, coeffs in zip(params, self.ops, self.coeffs):
            op = self.operator_basis[op]
            term = np.sum(np.stack([c * kron(o) for o, c in zip(op, coeffs)]), axis=0)

            if isinstance(param, jax.Array):
                term = jnp.array(term, dtype=np.complex64)

            H = H + param * term if H is not None else param * term

        return H

    def build_sparse_hamiltonian(self, params: list):
        """
        Get the sparse matrix representation of the Hamiltonian

        Args:
            params (list[float]): the coefficients of the Hamiltonian;
                the list must have length self.num_parameters

        Returns:
            scipy.sparse.csr_array: the sparse matrix representation of the Hamiltonian
        """
        def kron(arr):
            arr = [scipy.sparse.csr_array(a) for a in arr]
            return functools.reduce(lambda a, b: scipy.sparse.kron(a, b, format='csr'), arr)

        H = scipy.sparse.csr_array((self.d ** self.n, self.d ** self.n))

        for param, op, coeffs in zip(params, self.ops, self.coeffs):
            op = self.operator_basis[op]
            term = sum([c * kron(o) for o, c in zip(op, coeffs)])
            H = H + param * term

        return H


class SU2Hamiltonian(Hamiltonian):
    """
    Abstract class for Hamiltonians over SU(2)
    """

    @property
    def d(self) -> int:
        """
        Get the dimension of the Hilbert space of each qudit. For
            SU(2) this is 2
        """
        return 2

    @property
    def operator_basis(self):
        """
        Get the operator basis of the Hilbert space of each qudit. For
            SU(2) this is just the Pauli matrices plus identity
        """
        return np.stack([
            np.eye(2),
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]]),
        ])


class SU3Hamiltonian(Hamiltonian):
    """
    Abstract class for Hamiltonians over SU(3)
    """

    @property
    def d(self) -> int:
        """
        Get the dimension of the Hilbert space of each qudit. For
            SU(3) this is 3
        """
        return 3

    @property
    def operator_basis(self):
        """
        Get the operator basis of the Hilbert space of each qudit. For
            SU(3) this is the Gell-Mann matrices plus identity
        """
        return np.stack([
            np.eye(3),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
            np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
            np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
            np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3)
        ])


class IsingHamiltonian(SU2Hamiltonian):
    """
    The homogeneous Ising Hamiltonian with no external field:
        :math:`H = a(X_1X_2 + X_2X_3 + \cdots + X_{n-1}X_n)`.
        This Hamiltonian has 1 parameter and n - 1 Pauli observables.
    """

    def __init__(self, n):
        """
        Create an Ising Hamiltonian.

        Args:
            n: number of qubits
        """
        pauli_ops = np.array([[0] * i + [1, 1] + [0] * (n - i - 2) for i in range(n - 1)])
        super().__init__(n, [pauli_ops])


class TransverseIsingHamiltonian(SU2Hamiltonian):
    """
    The transverse-field Ising Hamiltonian:
        :math:`H = a(X_1X_2 + X_2X_3 + \cdots + X_{n-1}X_n) + b(Z_1 + Z_2 + \cdots + Z_n)`.
        This Hamiltonian has 2 parameters and 2n - 1 Pauli observables.
    """

    def __init__(self, n):
        """
        Create a transverse field Ising Hamiltonian.

        Args:
            n: number of qubits
        """
        pauli_ops = np.array([[0] * i + [1, 1] + [0] * (n - i - 2) for i in range(n - 1)])
        field = np.array([[0] * i + [3] + [0] * (n - i - 1) for i in range(n)])
        super().__init__(n, [pauli_ops, field])


class HeisenbergHamiltonian(SU2Hamiltonian):
    """
    The homogeneous Heisenberg Hamiltonian with a transverse field. The transverse
        field takes different values for all spins:
        :math:`H = \sum_{i=1}^n aX_iX_{i+1} + bY_iY_{i+1} + cZ_iZ_{i+1} + d_iX_i`.
        This Hamiltonian has n + 3 parameters and 4n - 3 Pauli observables.
    """

    def __init__(self, n):
        """
        Create a Heisenberg Hamiltonian.

        Args:
            n: number of qubits
        """
        x_coupling = np.array([[0] * i + [1, 1] + [0] * (n - i - 2) for i in range(n - 1)])
        y_coupling = np.array([[0] * i + [2, 2] + [0] * (n - i - 2) for i in range(n - 1)])
        z_coupling = np.array([[0] * i + [3, 3] + [0] * (n - i - 2) for i in range(n - 1)])
        field = [np.array([[0] * i + [1] + [0] * (n - i - 1)]) for i in range(n)]
        super().__init__(n, [x_coupling] + [y_coupling] + [z_coupling] + field)


class HeteroHeisenbergHamiltonian(SU2Hamiltonian):
    """
    The heterogeneous Heisenberg Hamiltonian with a transverse field. The transverse
        field takes different values for all spins:
        :math:`H = \sum_{i=1}^n a_iX_iX_{i+1} + b_iY_iY_{i+1} + c_iZ_iZ_{i+1} + d_iX_i`.
        This Hamiltonian has 4n - 3 parameters and 4n - 3 Pauli observables.
    """

    def __init__(self, n):
        """
        Create a heterogeneous Heisenberg Hamiltonian.

        Args:
            n: number of qubits
        """
        pauli_ops = []
        pauli_ops.extend([np.array([[0] * i + [1, 1] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [2, 2] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [3, 3] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [1] + [0] * (n - i - 1)]) for i in range(n)])

        super().__init__(n, pauli_ops)


class PXPHamiltonian(SU2Hamiltonian):
    """
    The PXP Hamiltonian :math:`H=\sum_{i=1}^n a_iP_{i-1}X_iP_{i+1}`, where :math:`P_i` is
        the projector onto the :math:`|0\langle` state for the i-th site.
        This Hamiltonian has n parameters and 4n - 4 Pauli observables, and has the 
        property that it is non-ergodic.
    """

    def __init__(self, n):
        """
        Create a PXP Hamiltonian.

        Args:
            n: number of qubits
        """
        pauli_ops = []
        coeffs = []

        # left boundary
        pauli_ops.append(np.array([[1, 0] + [0] * (n - 2), [1, 3] + [0] * (n - 2)]))
        coeffs.append(np.array([0.5, 0.5]))

        # internal
        for i in range(n - 2):
            pauli_ops.append(np.array([[0] * i + [0, 1, 0] + [0] * (n - i - 3),
                                       [0] * i + [0, 1, 3] + [0] * (n - i - 3),
                                       [0] * i + [3, 1, 0] + [0] * (n - i - 3),
                                       [0] * i + [3, 1, 3] + [0] * (n - i - 3)]))
            coeffs.append(np.array([0.5, 0.5, 0.5, 0.5]))

        # right boundary
        pauli_ops.append(np.array([[0] * (n - 2) + [0, 1], [0] * (n - 2) + [3, 1]]))
        coeffs.append(np.array([0.5, 0.5]))

        super().__init__(n, pauli_ops, coeffs=coeffs)


class HeisenbergNNNHamiltonian(SU2Hamiltonian):
    """
    The anisotropic Heisenberg Hamiltonian with next-nearest-neighbours coupling
        and external field in any direction.
        This Hamiltonian has 9n - 9 parameters and 9n - 9 Pauli observables.
    """

    def __init__(self, n):
        """
        Create a heterogeneous Heisenberg Hamiltonian.

        Args:
            n: number of qubits
        """
        pauli_ops = []

        # nearest neighbour couplings
        pauli_ops.extend([np.array([[0] * i + [1, 1] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [2, 2] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [3, 3] + [0] * (n - i - 2)]) for i in range(n - 1)])

        # next nearest neighbour couplings
        pauli_ops.extend([np.array([[0] * i + [1, 0, 1] + [0] * (n - i - 3)]) for i in range(n - 2)])
        pauli_ops.extend([np.array([[0] * i + [2, 0, 2] + [0] * (n - i - 3)]) for i in range(n - 2)])
        pauli_ops.extend([np.array([[0] * i + [3, 0, 3] + [0] * (n - i - 3)]) for i in range(n - 2)])

        # external field
        pauli_ops.extend([np.array([[0] * i + [1] + [0] * (n - i - 1)]) for i in range(n)])
        pauli_ops.extend([np.array([[0] * i + [2] + [0] * (n - i - 1)]) for i in range(n)])
        pauli_ops.extend([np.array([[0] * i + [3] + [0] * (n - i - 1)]) for i in range(n)])

        super().__init__(n, pauli_ops)


class HeisenbergSSSHamiltonian(SU2Hamiltonian):
    """
    The anisotropic Heisenberg Hamiltonian with next-nearest-neighbours coupling
        and external field in any direction. This defers from HeisenbergNNNHamiltonian
        in that the coupling is XXX, YYY, and ZZZ instead of XIX, YIY, ZIZ.
        This Hamiltonian has 9n - 9 parameters and 9n - 9 Pauli observables.
    """

    def __init__(self, n):
        """
        Create a heterogeneous Heisenberg Hamiltonian.

        Args:
            n: number of qubits
        """
        pauli_ops = []

        # nearest neighbour couplings
        pauli_ops.extend([np.array([[0] * i + [1, 1] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [2, 2] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [3, 3] + [0] * (n - i - 2)]) for i in range(n - 1)])

        # next nearest neighbour couplings
        pauli_ops.extend([np.array([[0] * i + [1, 1, 1] + [0] * (n - i - 3)]) for i in range(n - 2)])
        pauli_ops.extend([np.array([[0] * i + [2, 2, 2] + [0] * (n - i - 3)]) for i in range(n - 2)])
        pauli_ops.extend([np.array([[0] * i + [3, 3, 3] + [0] * (n - i - 3)]) for i in range(n - 2)])

        # external field
        pauli_ops.extend([np.array([[0] * i + [1] + [0] * (n - i - 1)]) for i in range(n)])
        pauli_ops.extend([np.array([[0] * i + [2] + [0] * (n - i - 1)]) for i in range(n)])
        pauli_ops.extend([np.array([[0] * i + [3] + [0] * (n - i - 1)]) for i in range(n)])

        super().__init__(n, pauli_ops)


class DenseNNHamiltonian(SU2Hamiltonian):
    """
    The Hamiltonian containing all Pauli couplings on nearest-neighbour spins.
        This Hamiltonian has 12n - 9 parameters and 12n - 6 Pauli observables.
    """

    def __init__(self, n):
        """
        Create a dense nearest-neighbour Hamiltonian.

        Args:
            n: number of qubits
        """
        pauli_ops = []

        # external field
        pauli_ops.extend([np.array([[0] * i + [1] + [0] * (n - i - 1)]) for i in range(n)])
        pauli_ops.extend([np.array([[0] * i + [2] + [0] * (n - i - 1)]) for i in range(n)])
        pauli_ops.extend([np.array([[0] * i + [3] + [0] * (n - i - 1)]) for i in range(n)])

        # nearest neighbour couplings
        pauli_ops.extend([np.array([[0] * i + [1, 1] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [1, 2] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [1, 3] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [2, 1] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [2, 2] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [2, 3] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [3, 1] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [3, 2] + [0] * (n - i - 2)]) for i in range(n - 1)])
        pauli_ops.extend([np.array([[0] * i + [3, 3] + [0] * (n - i - 2)]) for i in range(n - 1)])

        super().__init__(n, pauli_ops)


class HeteroHeisenbergSU3Hamiltonian(SU3Hamiltonian):
    """
    The heterogeneous Heisenberg Hamiltonian with a transverse field. The transverse
        field takes different values for all spins:
        :math:`H = \sum_{i=1}^n a_iX_iX_{i+1} + b_iY_iY_{i+1} + c_iZ_iZ_{i+1} + d_iX_i`.
        This Hamiltonian has 4n - 3 parameters and 4n - 3 Pauli observables.
    """

    def __init__(self, n):
        """
        Create a heterogeneous Heisenberg Hamiltonian.

        Args:
            n: number of qubits
        """
        gellmann_ops = []
        gellmann_ops.extend([np.array([[0] * i + [1, 1] + [0] * (n - i - 2)]) for i in range(n - 1)])
        gellmann_ops.extend([np.array([[0] * i + [2, 2] + [0] * (n - i - 2)]) for i in range(n - 1)])
        gellmann_ops.extend([np.array([[0] * i + [3, 3] + [0] * (n - i - 2)]) for i in range(n - 1)])
        gellmann_ops.extend([np.array([[0] * i + [4, 4] + [0] * (n - i - 2)]) for i in range(n - 1)])
        gellmann_ops.extend([np.array([[0] * i + [5, 5] + [0] * (n - i - 2)]) for i in range(n - 1)])
        gellmann_ops.extend([np.array([[0] * i + [6, 6] + [0] * (n - i - 2)]) for i in range(n - 1)])
        gellmann_ops.extend([np.array([[0] * i + [7, 7] + [0] * (n - i - 2)]) for i in range(n - 1)])
        gellmann_ops.extend([np.array([[0] * i + [8, 8] + [0] * (n - i - 2)]) for i in range(n - 1)])
        gellmann_ops.extend([np.array([[0] * i + [1] + [0] * (n - i - 1)]) for i in range(n)])

        super().__init__(n, gellmann_ops)
