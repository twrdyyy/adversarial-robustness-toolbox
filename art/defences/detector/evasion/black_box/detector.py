from typing import Optional, Tuple, Union, List

from art.data_generators import DataGenerator
from art.estimators.classification.classifier import ClassifierNeuralNetwork

from collections import deque

import numpy as np
import logging

from sklearn.neighbors import NearestNeighbors

from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class BlackBoxDetector(ClassifierNeuralNetwork):
    """
    Stateful detection on Black-Box by Twardy F. (2021).

    | Paper link: https://arxiv.org/abs/1907.05587
    """

    def __init__(
            self,
            classifier,
            similarity_encoder,
            distance_function,
            detection_threshold,
            k_neighbors,
            initial_last_queries,
            max_memory_size,
            knn=None
    ) -> None:
        super().__init__(
            model=None,
            clip_values=classifier.clip_values,
            channels_first=classifier.channels_first,
            preprocessing_defences=classifier.preprocessing_defences,
            preprocessing=classifier.preprocessing,
        )

        self.detector = classifier

        self.similarity_encoder = similarity_encoder

        self.distance_function = distance_function

        self.detection_threshold = detection_threshold

        self.k_neighbors = k_neighbors

        # memory queue

        self.memory_queue = deque(initial_last_queries, maxlen=max_memory_size)

        if not knn:
            self.knn = NearestNeighbors(
                n_neighbors=self.k_neighbors,
                metrics=distance_function
            )
        else:
            self.knn = knn

    def scan(
            self,
            query,
            last_queries=None,
    ) -> Tuple[bool, float, float]:
        if last_queries:
            self.memory_queue.extend(last_queries)

        encoded_query = self.similarity_encoder.predict(query)
        encoded_memory = self.similarity_encoder.predict(np.array(self.memory_queue))

        self.knn.fit(encoded_memory)

        k_distances, _ = self.knn.kneighbors(encoded_query)

        mean_distance = np.mean(k_distances)

        self.memory_queue.append(query)

        return mean_distance < self.detection_threshold, mean_distance, self.detection_threshold

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the detector using training data. Assumes that the classifier is already trained.

        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform detection of adversarial data and return prediction as tuple.

        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the classifier using the generator gen that yields batches as specified. This function is not supported
        for this detector.

        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Loss values.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    @property
    def nb_classes(self) -> int:
        return self.detector.nb_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.detector.input_shape

    @property
    def clip_values(self) -> Optional["CLIP_VALUES_TYPE"]:
        return self.detector.clip_values

    @property
    def channels_first(self) -> bool:
        """
        :return: Boolean to indicate index of the color channels in the sample `x`.
        """
        return self.channels_first

    @property
    def classifier(self) -> ClassifierNeuralNetwork:
        """
        :return: Classifier.
        """
        return self.detector

    def class_gradient(  # pylint: disable=W0221
            self, x: np.ndarray, label: Union[int, List[int], None] = None, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        return self.detector.class_gradient(x=x, label=label, training_mode=training_mode, **kwargs)

    def loss_gradient(  # pylint: disable=W0221
            self, x: np.ndarray, y: np.ndarray, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of the same shape as `x`.
        """
        return self.detector.loss_gradient(x=x, y=y, training_mode=training_mode, **kwargs)

    def get_activations(
            self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`. This function is not supported for this detector.

        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str] = None) -> None:
        self.detector.save(filename, path)
