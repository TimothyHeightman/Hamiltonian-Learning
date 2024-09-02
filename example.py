import itertools
import pickle

from tqdm import tqdm
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt

from src.utils import integer_to_binary
from src.dataset import Dataset, ExactSimulationDatasetGenerator
from src.models import ExactModel, ODEModel, NeuralODEModel
from src.hamiltonian import HeteroHeisenbergHamiltonian
from src.loop import NLLLoop, NLLWeightDecayLoop


from jax import config
config.update("jax_enable_x64", True)


def run_robustness_experiment():
    n, num_states, num_times, num_paulis, shots = 6, 5, 5, 200, 100
    H = HeteroHeisenbergHamiltonian(n)

    print(f'Hamiltonian: {H.__class__.__name__}, Parameters {H.num_parameters}, Pauli ops {H.num_observables}')

    for trial in range(50):
        print(f'Running trial {trial}...')

        dataset_gen = ExactSimulationDatasetGenerator(H)
        train_set = dataset_gen.generate_random_dataset(num_states, num_times, num_paulis, shots, dt=0.2, seed=trial + 1000)

        (train_set.set_batch_times(batch_size=num_times)
                  .set_batch_pauli_obs(batch_size=num_paulis)
                  .set_states_processor(lambda state: np.eye(H.d ** n)[state].astype(np.complex64))
                  .set_samples_processor(lambda samples: samples[:, :, None])
                  .set_shuffle(True))

        true_params = train_set.true_params
        print('True parameters:', true_params)

        model = NeuralODEModel(H, key=jax.random.key(trial + 100000))

        # for this simple example just use a fixed learning rate instead of the curriculum learning scheme
        schedule = 0.01

        optimizer = optax.adam(learning_rate=schedule)

        print('Initial parameters:', model.get_hamiltonian_parameters())

        loop = NLLWeightDecayLoop(model, optimizer, l2=1e-3)
        loop.train(train_set, num_epochs=4)

        print('Learned parameters:', model.get_hamiltonian_parameters())

        loop.save_metrics(f'runs/node_{H.__class__.__name__}_{n}_{trial:03d}.pkl')
        loop.save_model(f'runs/node_{H.__class__.__name__}_{n}_{trial:03d}.eqx')


if __name__ == '__main__':
    run_robustness_experiment()

