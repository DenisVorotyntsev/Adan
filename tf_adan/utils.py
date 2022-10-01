from typing import List, Dict

import torch
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from tf_adan.adan import Adan
from tf_adan.torch_adan import Adan as TorchAdan


def get_adan_optimizers(
    learning_rate: float,
    beta_1: float,
    beta_2: float,
    beta_3: float,
    epsilon: float,
    weight_decay: float,
    params_to_opt: List[torch.tensor],
):
    tf_adan = Adan(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        beta_3=beta_3,
        epsilon=epsilon,
        weight_decay=weight_decay,
    )
    torch_adan = TorchAdan(
        params_to_opt,
        lr=learning_rate,
        betas=[beta_1, beta_2, beta_3],
        eps=epsilon,
        weight_decay=weight_decay,
    )

    # test Adam from torch and tf to compare
    # tf_adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # torch_adam = torch.optim.Adam(params_to_opt, lr=learning_rate)
    return tf_adan, torch_adan


def plot_loss_curves_from_results(results: Dict[str, any]) -> None:
    tf_loss_history = np.array(results["tf_loss_history"])
    torch_loss_history = np.array(results["torch_loss_history"])

    figsize = (16, 4)

    plt.figure(figsize=figsize)
    plt.title("Tf/torch loss history")
    plt.plot(tf_loss_history, label="tf")
    plt.plot(torch_loss_history, label="torch")
    plt.legend()
    plt.show()

    delta = tf_loss_history - torch_loss_history
    plt.figure(figsize=figsize)
    plt.title("Tf/torch delta loss history")
    plt.plot(delta)
    plt.show()

    ratio = 100 * abs(tf_loss_history - torch_loss_history) / torch_loss_history
    plt.figure(figsize=figsize)
    plt.title("Tf/torch ratio loss history (%)")
    plt.plot(ratio)
    plt.show()


def check_results_almost_equal(results: Dict[str, any]) -> None:
    rtol = 1e-4

    tf_loss_history = np.array(results["tf_loss_history"])
    torch_loss_history = np.array(results["torch_loss_history"])
    np.testing.assert_allclose(tf_loss_history, torch_loss_history, rtol=rtol)
    print("loss almost equal")

    tf_w = np.array(results["tf_w"])
    torch_w = np.array(results["torch_w"])
    np.testing.assert_allclose(tf_w, torch_w, rtol=rtol)
    print("weigths are almost equal")
