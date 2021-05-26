# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Feature Adversaries attack.

| Paper link: https://arxiv.org/abs/1511.05122
"""
import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer

    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class FeatureAdversariesPyTorch(EvasionAttack):
    """
    This class represent a Feature Adversaries evasion attack in PyTorch.

    | Paper link: https://arxiv.org/abs/1511.05122
    """

    attack_params = EvasionAttack.attack_params + [
        "optimizer",
        "optimizer_kwargs",
        "delta",
        "lambda_",
        "layer",
        "max_iter",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        optimizer: Optional["Optimizer"] = None,
        optimizer_kwargs: Optional[dict] = None,
        delta: float = 15 / 255,
        lambda_: float = 1.0,
        layer: Optional[Union[int, str]] = -1,
        max_iter: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        """
        Create a :class:`.FeatureAdversariesPyTorch` instance.

        :param classifier: A trained classifier.
        :param optimizer: Optimizer applied to problem (default: torch.optim.LBFGS).
        :param optimizer_kwargs: Additional optimizer arguments.
        :param delta: The maximum deviation between source and guide images.
        :param lambda_: Regularization parameter of the L-inf soft constraint.
        :param layer: Index of the representation layer.
        :param max_iter: The maximum number of iterations.
        :param batch_size: Batch size.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)

        if optimizer is None:
            from torch.optim import LBFGS

            self._optimizer = LBFGS
        self._optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self.delta = delta
        self.lambda_ = lambda_
        self.layer = layer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self._check_params()

    def _generate_batch(self, x: "Tensor", y: "Tensor") -> "Tensor":
        """
        Generate adversarial batchy.

        :param x: Source samples.
        :param y: Guide samples.
        :return: Batch of adversarial examples.
        """
        import torch

        adv = x.clone().detach().to(self.estimator.device)

        def loss_fn(source_orig, source_adv, guide):
            # # clip perturbation
            # perturbation = torch.clamp(adv - x, -self.delta, self.delta)
            # adv = x + perturbation

            adv_representation = self.estimator.get_activations(source_adv, self.layer, self.batch_size, True)
            guide_representation = self.estimator.get_activations(guide, self.layer, self.batch_size, True)

            dim = tuple(range(1, len(source_adv.shape)))
            soft_constraint = torch.amax(torch.abs(source_adv - source_orig), dim=dim)

            dim = tuple(range(1, len(adv_representation.shape)))
            representation_loss = torch.sum(torch.square(adv_representation - guide_representation), dim=dim)

            loss = torch.mean(representation_loss + self.lambda_ * soft_constraint)
            return loss

        opt = self._optimizer(params=[adv], **self._optimizer_kwargs)

        for i in trange(self.max_iter, desc="Adversarial Patch TensorFlow v2", disable=not self.verbose):
            adv.requires_grad = True

            def closure():
                if torch.is_grad_enabled():
                    opt.zero_grad()
                loss = loss_fn(x, adv, y)
                if loss.requires_grad:
                    loss.backward()
                return loss

            opt.step(closure)

            # with torch.no_grad():
            #     # clip perturbation
            #     perturbation = torch.clamp(adv - x, -self.delta, self.delta)
            #     adv = x + perturbation
            #     if self.estimator.clip_values is not None:
            #         adv = adv.clamp(*self.estimator.clip_values)

            # TODO remove block later
            with torch.no_grad():
                dim = tuple(range(1, len(adv.shape)))
                soft_constraint_tmp = torch.mean(torch.amax(torch.abs(adv - x), dim=dim)).item()

                loss_tmp = loss_fn(x, adv, y).item()
            print(f"Iter {i}, loss {loss_tmp}, constraint {soft_constraint_tmp}")

        return adv.detach().cpu()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: Source samples.
        :param y: Guide samples.
        :return: Adversarial examples.
        """
        import torch

        if y is None:
            raise ValueError("The value of guide `y` cannot be None. Please provide a `np.ndarray` of guide inputs.")
        if x.shape != y.shape:
            raise ValueError("The shape of source `x` and guide `y` must be of same shape.")
        if x.shape[1:] != self.estimator.input_shape:
            raise ValueError("Source and guide inputs must match `input_shape` of estimator.")

        nb_samples = x.shape[0]

        x_adversarial = [None] * nb_samples

        nb_batches = int(np.ceil(nb_samples / float(self.batch_size)))
        for m in range(nb_batches):
            # batch indices
            begin, end = m * self.batch_size, min((m + 1) * self.batch_size, nb_samples)

            # create batch of adversarial examples
            source_batch = torch.tensor(x[begin:end])
            guide_batch = torch.tensor(y[begin:end])
            x_adversarial[begin:end] = self._generate_batch(source_batch, guide_batch).numpy()
        return np.array(x_adversarial, dtype=x.dtype)

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        if not isinstance(self.delta, float):
            raise ValueError("The value of delta must be of type float.")
        if self.delta <= 0:
            raise ValueError("The maximum deviation value delta has to be positive.")

        if not isinstance(self.lambda_, float):
            raise ValueError("The value of lambda_ must be of type float.")
        if self.lambda_ < 0.0:
            raise ValueError("The regularization parameter `lambda_` has to be nonnegative.")

        if not isinstance(self.layer, int) and not isinstance(self.layer, str):
            raise ValueError("The value of the representation layer must be integer or string.")

        if not isinstance(self.max_iter, int):
            raise ValueError("The value of max_iter must be of type int.")
        if self.max_iter <= 0:
            raise ValueError("The maxium number of iterations has to be a positive.")

        if self.batch_size <= 0:
            raise ValueError("The batch size has to be positive.")
