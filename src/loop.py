import abc
import os
import pickle

from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt


class Callback(abc.ABC):

    def on_epoch_start(self, epoch):
        return

    def on_epoch_end(self, epoch):
        return

    def on_batch_start(self, epoch, step):
        return

    def on_batch_end(self, epoch, step):
        return


class Loop:

    def __init__(self, model, optimizer):
        """
        Create a Loop instance

        Args:
            model (ParametrizedModel): the model to use; this model must take as
                input the initial states, the timestamps, the Pauli observables,
                and the samples.
            optimizer (torch.optim.Optimizer): the optimizer to use
            dataset (Dataset): the dataset to use
            log_dir (str): directly to save the metrics
        """
        self.model = model
        self.optimizer = optimizer

        self.metric_fns = {}
        self.metrics = {'loss': []}

        for i in range(self.model.H.num_parameters):
            self.metrics[f'param{i:04d}'] = []
            self.metrics[f'true_param{i:04d}'] = None

    def plot_loss(self):
        plt.plot(np.arange(len(self.metrics['loss'])), self.metrics['loss'])
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

    def plot_params(self):
        for i in range(self.model.H.num_parameters):
            hist = self.metrics[f'param{i:04d}']
            line = plt.plot(np.arange(len(hist)), hist)[0]
            plt.axhline(self.metrics[f'true_param{i:04d}'], color=line.get_color(), linestyle='--')

        plt.xlabel("Iteration")
        plt.ylabel("Parameter Values")
        plt.show()

    @abc.abstractmethod
    def loss_fn(self, inputs, outputs):
        """
        Return the loss value evaluated on the given data
        """
        raise NotImplementedError

    def add_metric(self, fn, name):
        """
        Add a metric to log. The loss and the model parameters are logged by default.

        Args:
            fn (callable): a callable that takes as input the model outputs and
                returns a scalar
            name (str): the name of the metric
        """
        self.metric_fns[name] = fn
        self.metrics[name] = []

    def _log_metrics(self, loss, grads, outputs, dataset, step):
        """
        Log the given metrics
        """
        params = np.array(self.model.get_hamiltonian_parameters())
        grads = np.array(grads.get_hamiltonian_parameters())

        self.metrics['loss'].append(loss)
        
        for i, param in enumerate(params):
            self.metrics[f'param{i:04d}'].append(param)
            self.metrics[f'true_param{i:04d}'] = dataset.true_params[i]

        for name, fn in self.metric_fns.items():
            value = np.array(fn(outputs))
            self.metrics[name].append(value)

    def save_metrics(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.metrics, f)

    def save_model(self, filename):
        eqx.tree_serialise_leaves(filename, self.model)

    def load_model(self, filename):
        self.model = eqx.tree_deserialise_leaves(filename, self.model)

    def train(self, train_set, num_epochs, callbacks=None, **kwargs):
        """
        Train the model for the given number of epochs.

        Args:
            num_epochs (int): the number of epochs
            kwargs (dict): keyword arguments to pass to the model forward method
        """
        if callbacks is None:
            callbacks = []

        @eqx.filter_value_and_grad
        def compute_loss(model, states, times, pauli_obs, samples):
            outputs = model(states, times, pauli_obs, samples, **kwargs)
            loss = self.loss_fn((states, times, pauli_obs, samples), outputs)
            return loss

        @eqx.filter_jit
        def do_step(model, states, times, pauli_obs, samples, opt_state):
            loss, grads = compute_loss(model, states, times, pauli_obs, samples)
            params = eqx.filter(model, eqx.is_array)
            updates, opt_state = self.optimizer.update(grads, opt_state, params=params)
            model = eqx.apply_updates(model, updates)

            return loss, grads, model, opt_state

        opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

        for epoch in range(num_epochs):

            for callback in callbacks:
                callback.on_epoch_start(epoch)

            step = 0
            epoch_loop = tqdm(train_set)
            for states, times, pauli_obs, samples in epoch_loop:

                for callback in callbacks:
                    callback.on_batch_start(epoch, step)

                states, times, pauli_obs, samples = jnp.array(states), jnp.array(times), jnp.array(pauli_obs), jnp.array(samples)

                times = np.array(times).tolist()

                loss, grads, self.model, opt_state = do_step(self.model, states, times, pauli_obs, samples, opt_state)
                loss = loss.item()

                non_params_norm = sum([jnp.sqrt(jnp.sum(jnp.abs(p) ** 2)) for p in self.model.get_non_hamiltonian_parameters()])

                self._log_metrics(loss, grads, None, train_set, epoch * len(train_set) + step)
                epoch_loop.set_description(f"Epoch {epoch + 1}/{num_epochs}, Step {step + 1}/{len(train_set)}, "
                                           f"Loss {self.metrics['loss'][-1]:.6f}, Params {self.metrics['param0000'][-1]:.4f} {self.metrics['param0001'][-1]:.4f}, "
                                           f"MLP {non_params_norm:.5f}")

                for callback in callbacks:
                    callback.on_batch_end(epoch, step)

                step += 1

            for callback in callbacks:
                callback.on_epoch_end(epoch)


class NLLLoop(Loop):

    def loss_fn(self, _, probs):
        """
        Return the loss value evaluated on the given data
        """
        loss = -jnp.mean(jnp.log(probs))
        return loss


class KLDivLoop(Loop):

    def loss_fn(self, inputs, all_probs):
        samples = inputs[-1]
        num_times, num_paulis, shots = all_probs.shape[:3]

        kl_divs = []
        for i in range(num_times):
            kl_divs_times = []
            for j in range(num_paulis):
                _, indices, counts = jnp.unique(samples[i, j], return_index=True, return_counts=True, axis=0, size=shots)
                probs = all_probs[i, j][indices]
                target_probs = counts.astype(probs.dtype) / shots

                kl = jnp.sum(target_probs * jnp.log(jnp.where(target_probs == 0, 1, target_probs) / (probs + 1e-6)))
                kl_divs_times.append(kl)

            kl_divs.append(jnp.stack(kl_divs_times))

        return jnp.mean(jnp.stack(kl_divs))


class NLLWeightDecayLoop(NLLLoop):

    def __init__(self, model, optimizer, l2=0.1):
        super().__init__(model, optimizer)
        self.l2 = l2

    def train(self, train_set, num_epochs, callbacks=None, **kwargs):
        """
        Train the model for the given number of epochs.

        Args:
            num_epochs (int): the number of epochs
            kwargs (dict): keyword arguments to pass to the model forward method
        """

        class WeightDecayCallback(Callback):

            def on_batch_end(cls, epoch, step):
                new_params = []
                for param in self.model.get_non_hamiltonian_parameters():
                    new_params.append(param - self.l2 * param)

                self.model = eqx.tree_at(type(self.model).get_non_hamiltonian_parameters, self.model, new_params)

        callbacks = [] if callbacks is None else callbacks
        return super().train(train_set, num_epochs, callbacks + [WeightDecayCallback()], **kwargs)
