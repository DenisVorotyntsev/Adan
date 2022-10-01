from typing import List, Tuple, Dict

import torch
import tensorflow as tf
import numpy as np

from tf_adan.utils import get_adan_optimizers


STATE = np.random.RandomState(123)


def _generate_dense_data(
    n_batches: int = 5, n_samples: int = 100, n_features: int = 20, n_targets: int = 5
) -> Tuple[List[List[List[float]]], List[List[List[float]]]]:
    X = []
    Y = []
    for _ in range(n_batches):
        x = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples, n_targets)
        X.append(x.tolist())
        Y.append(y.tolist())
    return X, Y


def _generate_init_dense_w(
    n_features: int = 20, n_targets: int = 5
) -> List[List[float]]:
    w = np.random.rand(n_features, n_targets).tolist()
    return w


def _test_dense(
    learning_rate: float = 0.01,
    beta_1: float = 0.99,
    beta_2: float = 0.92,
    beta_3: float = 0.9,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0,
    n_epochs: int = 100,
) -> Dict[str, any]:
    results = {}

    # get dense data to training
    X, Y = _generate_dense_data()
    w = _generate_init_dense_w()

    results["x_before_training"] = X
    results["y_before_training"] = Y
    results["w_before_training"] = w

    # convert to tf/torch tensors
    torch_X = [torch.tensor(x, dtype=torch.float32, requires_grad=False) for x in X]
    torch_Y = [torch.tensor(y, dtype=torch.float32, requires_grad=False) for y in Y]
    torch_w = torch.tensor(w, dtype=torch.float32, requires_grad=True)

    tf_X = [tf.Variable(x, dtype=tf.float32, trainable=False) for x in X]
    tf_Y = [tf.Variable(y, dtype=tf.float32, trainable=False) for y in Y]
    tf_w = tf.Variable(w, dtype=tf.float32, trainable=True)

    # define functions to optimize
    def torch_function_to_optimize(x, y, w):
        y_hat = x @ w
        return torch.mean((y - y_hat) ** 2)

    def tensorflow_function_to_optimize(x, y, w):
        y_hat = x @ w
        return tf.reduce_mean((y - y_hat) ** 2)

    # get tf and torch optimizers with specified params
    tf_adan, torch_adan = get_adan_optimizers(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        beta_3=beta_3,
        epsilon=epsilon,
        weight_decay=weight_decay,
        params_to_opt=[torch_w],
    )

    # run torch optimization
    torch_loss_history = []
    for _ in range(n_epochs):
        for torch_x, torch_y in zip(torch_X, torch_Y):
            loss = torch_function_to_optimize(x=torch_x, y=torch_y, w=torch_w)
            torch_loss_history.append(loss.detach().numpy())
            torch_adan.zero_grad()
            loss.backward()
            torch_adan.step()

    # run tf optimization
    tf_loss_history = []
    for _ in range(n_epochs):
        for tf_x, tf_y in zip(tf_X, tf_Y):
            with tf.GradientTape() as tape:
                loss = tensorflow_function_to_optimize(x=tf_x, y=tf_y, w=tf_w)
                tf_loss_history.append(loss.numpy())
            grads = tape.gradient(loss, [tf_w])
            tf_adan.apply_gradients(zip(grads, [tf_w]))

    results.update(
        {
            "torch_w": torch_w.detach().numpy(),
            "tf_w": tf_w.numpy(),
            "torch_loss_history": torch_loss_history,
            "tf_loss_history": tf_loss_history,
        }
    )
    return results
