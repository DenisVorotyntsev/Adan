import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="tf_adan")
class Adan(tf.keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.98,
        beta_2: float = 0.92,
        beta_3: float = 0.99,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        name="Adan",
        **kwargs
    ):
        """
        Unofficial Adan optimizer implementation.

        See the paper for details - "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"
        https://arxiv.org/abs/2208.06677

        Args:
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            beta_1 (float, optional): beta1. Defaults to 0.98.
            beta_2 (float, optional): beta2. Defaults to 0.92.
            beta_3 (float, optional): beta3. Defaults to 0.99.
            epsilon (float, optional): a small constant for numerical stabilit. Defaults to 1e-8.
            weight_decay (float, optional): weight decay. Defaults to 0.0.
            name (str, optional): optimizer name. Defaults to "Adan".
        """
        super().__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("beta_3", beta_3)
        self._set_hyper("epsilon", epsilon)
        self._set_hyper("weight_decay", weight_decay)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "exp_avg", initializer="zeros")
        for var in var_list:
            self.add_slot(var, "exp_avg_diff", initializer="zeros")
        for var in var_list:
            self.add_slot(var, "exp_avg_sq", initializer="zeros")
        for var in var_list:
            self.add_slot(var, "prev_grad", initializer="zeros")
        for var in var_list:
            self.add_slot(
                var, "is_grad_step_made", initializer="zeros"
            )  # needed for sparse grads

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)

        lr = self._get_hyper("learning_rate", var_dtype)
        weight_decay = tf.identity(self._get_hyper("weight_decay", var_dtype))
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        beta_3_t = tf.identity(self._get_hyper("beta_3", var_dtype))

        apply_state[(var_device, var_dtype)].update(
            dict(
                step=local_step,
                lr=lr,
                weight_decay=weight_decay,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_2_t=beta_2_t,
                beta_3_t=beta_3_t,
                bias_correction1=1.0 - tf.pow(beta_1_t, local_step),
                bias_correction2=1.0 - tf.pow(beta_2_t, local_step),
                bias_correction3=1.0 - tf.pow(beta_3_t, local_step),
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # get coefs on the current step
        step = coefficients["step"]
        lr = coefficients["lr"]
        beta1 = coefficients["beta_1_t"]
        beta2 = coefficients["beta_2_t"]
        beta3 = coefficients["beta_3_t"]
        bias_correction1 = coefficients["bias_correction1"]
        bias_correction2 = coefficients["bias_correction2"]
        bias_correction3 = coefficients["bias_correction3"]
        weight_decay = coefficients["weight_decay"]
        eps = coefficients["epsilon"]

        # get params
        exp_avg = self.get_slot(var, "exp_avg")
        exp_avg_sq = self.get_slot(var, "exp_avg_sq")
        exp_avg_diff = self.get_slot(var, "exp_avg_diff")

        # set prev grad to current grad where steps = 0
        is_grad_step_made = self.get_slot(var, "is_grad_step_made")
        prev_grad = tf.where(
            is_grad_step_made == 0,
            tf.identity(grad),
            tf.identity(self.get_slot(var, "prev_grad")),
        )

        # update steps
        _ = self.get_slot(var, "is_grad_step_made").assign(
            tf.ones_like(is_grad_step_made), use_locking=self._use_locking
        )

        # calc params on step t
        diff = grad - prev_grad
        update = grad + beta2 * diff
        exp_avg = exp_avg * beta1 + grad * (1 - beta1)
        exp_avg_diff = exp_avg_diff * beta2 + diff * (1 - beta2)
        exp_avg_sq = exp_avg_sq * beta3 + update**2 * (1 - beta3)

        # update params
        _ = self.get_slot(var, "exp_avg").assign(exp_avg, use_locking=self._use_locking)
        _ = self.get_slot(var, "exp_avg_diff").assign(
            exp_avg_diff, use_locking=self._use_locking
        )
        _ = self.get_slot(var, "exp_avg_sq").assign(
            exp_avg_sq, use_locking=self._use_locking
        )

        # update prev gradient
        _ = self.get_slot(var, "prev_grad").assign(grad, use_locking=self._use_locking)

        # calc var update
        denom = (exp_avg_sq / bias_correction3) ** 0.5 + eps
        var_update = (
            (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2)
        ) / denom

        var_updated = var - var_update * lr
        var_updated = var_updated / (1 + lr * weight_decay)
        return var.assign(var_updated, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # get coefs on the current step
        step = coefficients["step"]
        lr = coefficients["lr"]
        beta1 = coefficients["beta_1_t"]
        beta2 = coefficients["beta_2_t"]
        beta3 = coefficients["beta_3_t"]
        bias_correction1 = coefficients["bias_correction1"]
        bias_correction2 = coefficients["bias_correction2"]
        bias_correction3 = coefficients["bias_correction3"]
        weight_decay = coefficients["weight_decay"]
        eps = coefficients["epsilon"]

        # get params
        exp_avg = tf.gather(self.get_slot(var, "exp_avg"), indices)
        exp_avg_sq = tf.gather(self.get_slot(var, "exp_avg_sq"), indices)
        exp_avg_diff = tf.gather(self.get_slot(var, "exp_avg_diff"), indices)
        prev_grad = tf.gather(self.get_slot(var, "prev_grad"), indices)
        is_grad_step_made = tf.gather(self.get_slot(var, "is_grad_step_made"), indices)

        # set prev grad to current grad where steps = 0
        prev_grad = tf.where(
            is_grad_step_made == 0, tf.identity(grad), tf.identity(prev_grad)
        )

        # update steps
        with tf.control_dependencies([is_grad_step_made]):
            is_grad_step_made = self._resource_scatter_update(
                self.get_slot(var, "is_grad_step_made"), indices, 1.0
            )

        # calc params on step t
        diff = grad - prev_grad
        update = grad + beta2 * diff
        exp_avg = exp_avg * beta1 + grad * (1 - beta1)
        exp_avg_diff = exp_avg_diff * beta2 + diff * (1 - beta2)
        exp_avg_sq = exp_avg_sq * beta3 + update**2 * (1 - beta3)

        # update params
        with tf.control_dependencies([exp_avg]):
            exp_avg = self._resource_scatter_update(
                self.get_slot(var, "exp_avg"), indices, exp_avg
            )
        with tf.control_dependencies([exp_avg_diff]):
            exp_avg_diff = self._resource_scatter_update(
                self.get_slot(var, "exp_avg_diff"), indices, exp_avg_diff
            )
        with tf.control_dependencies([exp_avg_sq]):
            exp_avg_sq = self._resource_scatter_update(
                self.get_slot(var, "exp_avg_sq"), indices, exp_avg_sq
            )

        # update prev grad
        with tf.control_dependencies([prev_grad]):
            prev_grad = self._resource_scatter_update(
                self.get_slot(var, "prev_grad"), indices, grad
            )

        # calc var update
        denom = (exp_avg_sq / bias_correction3) ** 0.5 + eps
        var_update = (
            (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2)
        ) / denom

        var_updated = var - var_update * lr
        var_updated = var_updated / (1 + lr * weight_decay)
        return tf.group(
            *[
                var.assign(var_updated, use_locking=self._use_locking),
                exp_avg,
                exp_avg_diff,
                exp_avg_sq,
                prev_grad,
                is_grad_step_made,
            ]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "beta_3": self._serialize_hyperparameter("beta_3"),
                "epsilon": self._serialize_hyperparameter("epsilon"),
            }
        )
        return config
