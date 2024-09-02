import abc
import itertools
from string import ascii_letters as ABC
import functools

import numpy as np
import scipy
import h5py

from .utils import kron, integer_to_binary


class Dataset:
    """
    A class that holds a dataset. Each dataset contains some initial states,
        evolution times, Pauli observables, and the observed samples.
    """

    def __init__(self, n, true_params, initial_states, times, pauli_obs, samples):
        self.n = n
        self.true_params = true_params
        self.initial_states = initial_states
        self.times = times
        self.pauli_obs = pauli_obs
        self.samples = samples

        self.batch_states = 1
        self.batch_times = 1
        self.batch_pauli_obs = 1
        self.batch_samples = 1

        self.states_processor = None
        self.times_processor = None
        self.paulis_processor = None
        self.samples_processor = None

        self.shuffle = False

    def set_batch_states(self, batch_size):
        """
        Set the batch size of the initial states

        Args:
            batch_size (int): the batch size

        Returns:
            Dataset: the modified dataset class for method chaining
        """
        self.batch_states = batch_size
        return self

    def set_batch_times(self, batch_size):
        """
        Set the batch size of the evolution times

        Args:
            batch_size (int): the batch size

        Returns:
            Dataset: the modified dataset class for method chaining
        """
        self.batch_times = batch_size
        return self

    def set_batch_pauli_obs(self, batch_size):
        """
        Set the batch size of the Pauli observables

        Args:
            batch_size (int): the batch size

        Returns:
            Dataset: the modified dataset class for method chaining
        """
        self.batch_pauli_obs = batch_size
        return self

    def set_batch_samples(self, batch_size):
        """
        Set the batch size of the samples

        Args:
            batch_size (int): the batch size

        Returns:
            Dataset: the modified dataset class for method chaining
        """
        self.batch_samples = batch_size
        return self

    def set_states_processor(self, fn):
        """
        Set the function to process the states during dataset iteration.

        Args:
            fn (callable): a callable with signature (np.ndarray,) -> np.ndarray,
                which takes as input a batch of initial states and returns another
                array containing the processed states

        Returns:
            Dataset: the modified dataset class for method chaining
        """
        self.states_processor = fn
        return self

    def set_times_processor(self, fn):
        """
        Set the function to process the times during dataset iteration.

        Args:
            fn (callable): a callable with signature (np.ndarray,) -> np.ndarray,
                which takes as input a batch of timestamps and returns another
                array containing the processed times

        Returns:
            Dataset: the modified dataset class for method chaining
        """
        self.times_processor = fn
        return self

    def set_paulis_processor(self, fn):
        """
        Set the function to process the Pauli observables during dataset iteration.

        Args:
            fn (callable): a callable with signature (np.ndarray,) -> np.ndarray,
                which takes as input a batch of Pauli observables and returns another
                array containing the processed observables

        Returns:
            Dataset: the modified dataset class for method chaining
        """
        self.paulis_processor = fn
        return self

    def set_samples_processor(self, fn):
        """
        Set the function to process the samples during dataset iteration.

        Args:
            fn (callable): a callable with signature (np.ndarray,) -> np.ndarray,
                which takes as input a batch of measurement samples and returns another
                array containing the processed samples

        Returns:
            Dataset: the modified dataset class for method chaining
        """
        self.samples_processor = fn
        return self

    def set_shuffle(self, shuffle):
        """
        Set whether to shuffle the dataset or not

        Args:
            shuffle (bool): True if dataset should be shuffled, and False otherwise

        Returns:
            Dataset: the modified dataset class for method chaining
        """
        self.shuffle = shuffle
        return self

    @staticmethod
    def load(filename):
        """
        Load a dataset from an HDF5 file.

        Args:
            filename (str): the name of the file to load from

        Returns:
            Dataset: the loaded dataset
        """
        with h5py.File(filename, 'r') as f:
            metadata = f.attrs
            n = metadata['n']
            true_params = metadata['true_params']

            initial_states = f['states'][:]
            times = f['times'][:]
            pauli_obs = f['pauli_obs'][:]
            samples = f['samples'][:]

        return Dataset(n, true_params, initial_states, times, pauli_obs, samples)

    def save(self, filename):
        """
        Save this dataset to an HDF5 file.

        Args:
            filename (str): the name of the file to save to
        """
        with h5py.File(filename, 'w') as f:
            metadata = f.attrs
            metadata['n'] = self.n
            metadata['true_params'] = self.true_params

            f.create_dataset('states', data=self.initial_states)
            f.create_dataset('times', data=self.times)
            f.create_dataset('pauli_obs', data=self.pauli_obs)
            f.create_dataset('samples', data=self.samples)

    def __len__(self):
        """
        Return the size of the dataset
        """
        num_state_batches = int(np.ceil(self.initial_states.shape[0] / self.batch_states))
        num_time_batches = int(np.ceil(self.times.shape[0] / self.batch_times))
        num_pauli_batches = int(np.ceil(self.pauli_obs.shape[0] / self.batch_pauli_obs))
        num_sample_batches = int(np.ceil(self.samples.shape[-1] / self.batch_samples))

        return num_state_batches * num_time_batches * num_pauli_batches * num_sample_batches

    def __iter__(self):
        """
        Return a generator that allows iteration through the dataset
        """
        num_state_batches = int(np.ceil(self.initial_states.shape[0] / self.batch_states))
        num_time_batches = int(np.ceil(self.times.shape[0] / self.batch_times))
        num_pauli_batches = int(np.ceil(self.pauli_obs.shape[0] / self.batch_pauli_obs))
        num_sample_batches = int(np.ceil(self.samples.shape[-1] / self.batch_samples))

        state_indices = np.arange(self.initial_states.shape[0])
        time_indices = np.arange(self.times.shape[0])
        pauli_indices = np.arange(self.pauli_obs.shape[0])

        batch_indices = list(itertools.product(np.arange(num_state_batches),
                                               np.arange(num_time_batches), 
                                               np.arange(num_pauli_batches),
                                               np.arange(num_sample_batches)))

        if self.shuffle:
            np.random.shuffle(state_indices)
            np.random.shuffle(time_indices)
            np.random.shuffle(pauli_indices)

            np.random.shuffle(batch_indices)

        for i, j, k, l in batch_indices:
            state_batch = state_indices[i * self.batch_states : (i + 1) * self.batch_states]
            time_batch = np.sort(time_indices[j * self.batch_times : (j + 1) * self.batch_times])
            pauli_batch = pauli_indices[k * self.batch_pauli_obs : (k + 1) * self.batch_pauli_obs]

            states = np.squeeze(self.initial_states[state_batch])
            times = np.squeeze(self.times[time_batch])
            paulis = np.squeeze(self.pauli_obs[pauli_batch])
        
            samples = np.squeeze(self.samples[
                state_batch[:, None, None], 
                time_batch[None, :, None],
                pauli_batch[None, None, :],
                l * self.batch_samples : (l + 1) * self.batch_samples
            ])

            if self.states_processor is not None:
                states = self.states_processor(states)

            if self.times_processor is not None:
                times = self.times_processor(times)

            if self.paulis_processor is not None:
                paulis = self.paulis_processor(paulis)

            if self.samples_processor is not None:
                samples = self.samples_processor(samples)

            yield states, times, paulis, samples


class DatasetGenerator:
    """
    A class to generate Datasets
    """

    def __init__(self, hamiltonian):
        """
        Create a DatasetGenerator object

        Args:
            hamiltonian (Hamiltonian): the Hamiltonian of the system
        """
        self.n = hamiltonian.n
        self.H = hamiltonian
        self.true_params = None

    def generate_states(self, num_states, rng):
        """
        Generate a random batch of initial states to use for evolution

        Args:
            num_states (int): the number of initial states
            rng (np.RandomGenerator): the numpy random generator to use

        Returns:
            list[np.ndarray]: the generated states
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evolve(self, params, initial_state, t):
        """
        Evolve the initial state under the Hamiltonian H for some time t, 
            using the provided Hamiltonian parameters.

        Args:
            params (np.ndarray): the Hamiltonian parameters
            initial_state (np.ndarray): the initial state with shape (2 ** n,)
            t (float): the time to evolve for

        Returns:
            np.ndarray: the final state after time evolution
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_observable(self, state, pauli_op, shots, rng):
        """
        Sample an observable on the given state.

        Args:
            state (np.ndarray): the final state to sample, which must be a value
                returned from the self.evolve method
            pauli_op (np.ndarray): the Pauli observable to sample; must have shape (n,)
                and contain either 0, 1, and 2, for X, Y, and Z respectively
            shots (int): the number of times to sample
            rng (np.RandomGenerator): the numpy random generator to use

        Returns:
            np.ndarray: the sample results of shape (shots,), representing indices
                into the computational basis
        """
        raise NotImplementedError

    def generate_dataset(self, initial_states, times, pauli_obs, shots, rng):
        """
        Generate a dataset containing measurement results (as bitstrings) of
            the provided Pauli observables. The measurement is performed on the
            final state after evolving under the Heisenberg Hamiltonian for all
            times in the array t.
        """
        indices = []

        for init_state in initial_states:
            init_state_indices = []
            for t in times:
                final_state = self.evolve(self.true_params, init_state, t)
                init_state_indices.append(np.stack([self.sample_observable(final_state, ob, shots, rng)
                                                    for ob in pauli_obs]))
            indices.append(np.stack(init_state_indices))

        return Dataset(self.n, self.true_params, initial_states, times, pauli_obs, np.stack(indices))

    def generate_random_dataset(self, num_initial_states, num_times, num_paulis, num_shots, dt=0.2, seed=None):
        rng = np.random.default_rng(seed)

        if self.true_params is None:
            self.true_params = rng.uniform(-1, 1, size=self.H.num_parameters)

        initial_states = self.generate_states(num_initial_states, rng)
        times = dt + dt * np.arange(num_times)
        pauli_obs = rng.integers(self.H.operator_basis.shape[0] - 1, size=(num_paulis, self.n))

        return self.generate_dataset(initial_states, times, pauli_obs, num_shots, rng)


class ExactSimulationDatasetGenerator(DatasetGenerator):
    """
    Use an exact state-vector simulation to generate the dataset
    """

    def __state_index_to_vector(self, index):
        state = np.zeros(self.H.d ** self.n)
        state[index] = 1
        return state

    def generate_states(self, num_states, rng):
        """
        Generate a random batch of initial states to use for evolution

        Args:
            num_states (int): the number of initial states
            rng (np.RandomGenerator): the numpy random generator to use

        Returns:
            list[np.ndarray]: the generated states
        """
        # if only one state, use the all zero state
        if num_states == 1:
            return np.array([0])
        
        return rng.integers(self.H.d ** self.n, size=(num_states,))

    def evolve(self, params, initial_state, t):
        """
        Evolve the initial state under the Hamiltonian H for some time t, 
            using the provided Hamiltonian parameters.

        Args:
            params (np.ndarray): the Hamiltonian parameters
            initial_state (np.ndarray): the initial state with shape (2 ** n,)
            t (float): the time to evolve for

        Returns:
            np.ndarray: the final state after time evolution
        """
        initial_state = self.__state_index_to_vector(initial_state)

        # up to 10 qubits, use exact time-evolution
        if self.n <= 10:
            H = self.H.build_dense_hamiltonian(params)
            U = scipy.linalg.expm(-1j * t * H)
            return U @ initial_state

        def state_derivative(t, state):
            return -1j * H.dot(state)

        H = self.H.build_sparse_hamiltonian(params)

        initial_state = initial_state.astype(np.complex128)
        final_state = scipy.integrate.solve_ivp(state_derivative, (0, t), initial_state, rtol=1e-4, atol=1e-8).y[:, -1]
        final_state = final_state / np.linalg.norm(final_state)

        return final_state

    def sample_observable(self, state, pauli_op, shots, rng):
        """
        Sample an observable on the given state.

        Args:
            state (np.ndarray): the final state to sample, which must be a value
                returned from the self.evolve method
            pauli_op (np.ndarray): the Pauli observable to sample; must have shape (n,)
                and contain either 0, 1, and 2, for X, Y, and Z respectively
            shots (int): the number of times to sample
            rng (np.RandomGenerator): the numpy random generator to use

        Returns:
            np.ndarray: the sample results of shape (shots,), representing indices
                into the computational basis
        """
        basis = self.H.operator_basis[1:]
        _, u = np.linalg.eigh(basis)
        u = np.conj(np.transpose(u[:, :, ::-1], axes=(0, 2, 1)))

        rots = u[pauli_op]

        state = state.reshape([self.H.d] * self.n)
        for i, rot in enumerate(rots):
            state = np.moveaxis(np.tensordot(rot, state, axes=([1], [i])), 0, i)

        state_rotated = state.reshape(-1)

        probs = np.abs(state_rotated) ** 2
        samples = rng.choice(np.arange(self.H.d ** self.n), size=shots, p=probs).astype(np.int64)

        return samples
