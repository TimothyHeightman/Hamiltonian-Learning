import abc
import functools
from string import ascii_letters as ABC

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import diffrax

from .hamiltonian import Hamiltonian


class ParametrizedModel(eqx.Module):
    """
    A model that learns a parametrized representation of the Hamiltonian
    """
    H: Hamiltonian
    n: int
    d: int

    def __init__(self, hamiltonian: Hamiltonian):
        """
        Create a ParametrizedModel instance

        Args:
            hamiltonian: the Hamiltonian to model
        """
        super().__init__()

        self.H = hamiltonian
        self.n = hamiltonian.n
        self.d = hamiltonian.d

    @abc.abstractmethod
    def get_hamiltonian_parameters(self):
        """
        Get the learned parameters of the Hamiltonian

        Returns:
            list[torch.Tensor]: the Hamiltonian parameters
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_non_hamiltonian_parameters(self):
        raise NotImplementedError


class StateModel(eqx.Module):
    """
    A model that provides access to some time-evolved state
    """

    @abc.abstractmethod
    def get_time_evolved_state(self, initial_state, times):
        """
        Return the time-evolved state
        """
        raise NotImplementedError


class ExactModel(ParametrizedModel):
    """
    A simple model that uses exact time evolution to calculate likelihoods
    """
    params: jax.Array

    def __init__(self, hamiltonian: Hamiltonian, key):
        """
        Create a BasicModel instance

        Args:
            hamiltonian: the Hamiltonian to model
        """
        super().__init__(hamiltonian)
        # self.params = 0.1 * jax.random.normal(key, shape=(hamiltonian.num_parameters,))
        self.params = key

    def get_hamiltonian_parameters(self):
        """
        Get the learned parameters of the Hamiltonian

        Returns:
            list[torch.Tensor]: the Hamiltonian parameters
        """
        return self.params

    def get_non_hamiltonian_parameters(self):
        return []

    def __call__(self, initial_state, ts, pauli_obs, indices):
        """
        Perform a forward pass through the model by calculating probabilities
            of the given indices after measuring the given observables

        Args:
            t: the time to simulate Hamiltonian dynamics for
            pauli_obs: an int tensor of shape (batch, n) to measure. 0, 1, and 2
                correspond to X, Y, and Z respectively.
            indices: an int tensor of shape (batch, shots) representing the measurement
                results. Each element of the tensor is an integer between 0 and 2 ** n - 1
                representing the integer value of the binary bitstring.
        """
        # input time, pauli observable, and index
        # output probabilities

        H = self.H.build_dense_hamiltonian(self.params)

        all_probs = []

        for i, t in enumerate(ts):
            U = jax.scipy.linalg.expm(-1j * t * H)
            final_state = U[:, initial_state].reshape([2] * self.n)

            # pauli_obs: (batch, n)
            u = jnp.array(np.stack([
                np.array([[1, 1], [1, -1]]) / np.sqrt(2),
                np.array([[1, -1j], [1, 1j]]) / np.sqrt(2),
                np.eye(2)
            ]), dtype=np.complex64)

            # (batch, n, 2, 2)
            rots = u[pauli_obs]

            # 'bcde,afb,agc,ahd,aie->afghi'
            contraction_indices = [ABC[1:1 + self.n]] + [f'a{ABC[1 + self.n + i]}{ABC[1 + i]}' for i in range(self.n)]
            target_indices = f'a{ABC[1 + self.n:1 + 2 * self.n]}'
            rot_state = jnp.einsum(
                ','.join(contraction_indices) + '->' + target_indices,
                final_state,
                *[rots[:, i] for i in range(self.n)]
            )

            probs = jnp.abs(rot_state).reshape((-1, 2 ** self.n)) ** 2
            # rot_state: (batch, 2 ** n)
            # bits: (batch, shots)

            gather = jnp.tile(jnp.arange(probs.shape[0])[:, None], (1, indices.shape[-1]))
            out = probs[gather, indices[i]]

            all_probs.append(out)

        return jnp.stack(all_probs)


class EvolveODELayer(eqx.Module):
    """
    A layer that outputs the time derivative of a state given a Hamiltonian
    """
    H: Hamiltonian

    H_params: jax.Array

    def __init__(self, hamiltonian: Hamiltonian, key):
        super().__init__()

        input_size = hamiltonian.d ** hamiltonian.n

        self.H = hamiltonian
        self.H_params = 0.1 * jax.random.normal(key, shape=(hamiltonian.num_parameters,))

    def __call__(self, t, initial_state, args):
        # initial_state: shape (d ** n,)
        H = self.H.build_dense_hamiltonian(self.H_params)

        x = -1j * H @ initial_state
        return x


class ODEModel(ParametrizedModel, StateModel):
    """
    A model that uses a neural ODE to simulate time evolution
    """
    ode: EvolveODELayer

    def __init__(self, hamiltonian: Hamiltonian, *args, **kwargs):
        """
        Create a NeuralODEModel instance

        Args:
            hamiltonian: the Hamiltonian to model
            args (tuple): extra arguments to pass to the ODE layer
            kwargs (dict): extra keyword arguments to pass to the ODE layer
        """
        super().__init__(hamiltonian)
        self.ode = EvolveODELayer(hamiltonian, *args, **kwargs)

    def get_hamiltonian_parameters(self):
        """
        Get the learned parameters of the Hamiltonian

        Returns:
            list[torch.Tensor]: the Hamiltonian parameters
        """
        return self.ode.H_params

    def get_non_hamiltonian_parameters(self):
        return []

    def get_time_evolved_state(self, initial_state, times):
        """
        Return the time-evolved state
        """
        # times = torch.cat([torch.zeros(1), times])

        # final_state = odeint(self.ode, initial_state, times, **kwargs)[1:]
        # final_state = final_state / torch.sqrt(torch.sum(torch.abs(final_state) ** 2, dim=-1, keepdim=True))

        final_state = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ode),
            diffrax.Bosh3(),
            t0=0,
            t1=times[-1],
            dt0=None,
            y0=initial_state,
            saveat=diffrax.SaveAt(ts=times),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
        ).ys

        final_state = final_state / jnp.sqrt(jnp.sum(jnp.abs(final_state) ** 2, axis=-1))[:, None]

        return final_state

    def __call__(self, initial_state, ts, pauli_obs, samples):
        final_state = self.get_time_evolved_state(initial_state, ts).reshape([-1] + [self.d] * self.n)

        basis = self.H.operator_basis[1:]
        u = np.linalg.eigh(basis)[1]
        u = np.conj(np.transpose(u[:, :, ::-1], axes=(0, 2, 1)))
        u = jnp.array(u, dtype=np.complex64)

        # (batch, n, 2, 2)
        rots = u[pauli_obs]

        all_probs = []

        for t in range(len(ts)):
            # 'bcde,afb,agc,ahd,aie->afghi'
            contraction_indices = [ABC[1:1 + self.n]] + [f'a{ABC[1 + self.n + i]}{ABC[1 + i]}' for i in range(self.n)]
            target_indices = f'a{ABC[1 + self.n:1 + 2 * self.n]}'
            rot_state = jnp.einsum(
                ','.join(contraction_indices) + '->' + target_indices,
                final_state[t],
                *[rots[:, i] for i in range(self.n)],
                optimize='greedy'
            )

            probs = jnp.abs(rot_state).reshape((-1, self.d ** self.n)) ** 2
            # rot_state: (batch, 2 ** n)
            # bits: (batch, shots)

            gather = jnp.tile(jnp.arange(probs.shape[0])[:, None], (1, samples.shape[-1]))
            out = probs[gather, samples[t]]

            all_probs.append(out)

        return jnp.stack(all_probs)


class EvolveNeuralODELayer(eqx.Module):
    """
    A layer that outputs the time derivative of a state given a Hamiltonian
    """
    H: Hamiltonian

    H_params: jax.Array
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, hamiltonian: Hamiltonian, key):
        super().__init__()

        input_size = hamiltonian.d ** hamiltonian.n

        self.H = hamiltonian

        keys = jax.random.split(key, 3)

        self.H_params = 0.1 * jax.random.normal(keys[0], shape=(hamiltonian.num_parameters,))
        self.fc1 = eqx.nn.Linear(2 * input_size, 2 * input_size, key=keys[1])
        self.fc2 = eqx.nn.Linear(2 * input_size, 2 * input_size, key=keys[2])

    def __call__(self, t, initial_state, args):
        # initial_state: shape (d ** n,)
        H = self.H.build_dense_hamiltonian(self.H_params)

        x = jnp.concatenate([jnp.real(initial_state), jnp.imag(initial_state)], axis=0)
        x = jax.nn.relu(self.fc1(x))
        x = self.fc2(x)
        x = x[:x.shape[0] // 2] + 1j * x[x.shape[0] // 2:]

        x = -1j * (x + H @ initial_state)
        return x


class NeuralODEModel(ParametrizedModel, StateModel):
    """
    A model that uses a neural ODE to simulate time evolution
    """
    ode: EvolveNeuralODELayer

    def __init__(self, hamiltonian: Hamiltonian, *args, **kwargs):
        """
        Create a NeuralODEModel instance

        Args:
            hamiltonian: the Hamiltonian to model
            args (tuple): extra arguments to pass to the ODE layer
            kwargs (dict): extra keyword arguments to pass to the ODE layer
        """
        super().__init__(hamiltonian)
        self.ode = EvolveNeuralODELayer(hamiltonian, *args, **kwargs)

    def get_hamiltonian_parameters(self):
        """
        Get the learned parameters of the Hamiltonian

        Returns:
            list[torch.Tensor]: the Hamiltonian parameters
        """
        return self.ode.H_params

    def get_non_hamiltonian_parameters(self):
        return [self.ode.fc1.weight, self.ode.fc1.bias, self.ode.fc2.weight, self.ode.fc2.bias]

    def get_time_evolved_state(self, initial_state, times):
        """
        Return the time-evolved state
        """
        # times = torch.cat([torch.zeros(1), times])

        # final_state = odeint(self.ode, initial_state, times, **kwargs)[1:]
        # final_state = final_state / torch.sqrt(torch.sum(torch.abs(final_state) ** 2, dim=-1, keepdim=True))

        final_state = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ode),
            diffrax.Bosh3(),
            t0=0,
            t1=times[-1],
            dt0=None,
            y0=initial_state,
            saveat=diffrax.SaveAt(ts=times),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
        ).ys

        final_state = final_state / jnp.sqrt(jnp.sum(jnp.abs(final_state) ** 2, axis=-1))[:, None]

        return final_state

    def __call__(self, initial_state, ts, pauli_obs, samples):
        final_state = self.get_time_evolved_state(initial_state, ts).reshape([-1] + [self.d] * self.n)

        basis = self.H.operator_basis[1:]
        u = np.linalg.eigh(basis)[1]
        u = np.conj(np.transpose(u[:, :, ::-1], axes=(0, 2, 1)))
        u = jnp.array(u, dtype=np.complex64)

        # (batch, n, 2, 2)
        rots = u[pauli_obs]

        all_probs = []

        for t in range(len(ts)):
            # 'bcde,afb,agc,ahd,aie->afghi'
            contraction_indices = [ABC[1:1 + self.n]] + [f'a{ABC[1 + self.n + i]}{ABC[1 + i]}' for i in range(self.n)]
            target_indices = f'a{ABC[1 + self.n:1 + 2 * self.n]}'
            rot_state = jnp.einsum(
                ','.join(contraction_indices) + '->' + target_indices,
                final_state[t],
                *[rots[:, i] for i in range(self.n)],
                optimize='greedy'
            )

            probs = jnp.abs(rot_state).reshape((-1, self.d ** self.n)) ** 2
            # rot_state: (batch, 2 ** n)
            # bits: (batch, shots)

            gather = jnp.tile(jnp.arange(probs.shape[0])[:, None], (1, samples.shape[-1]))
            out = probs[gather, samples[t]]

            all_probs.append(out)

        return jnp.stack(all_probs)
