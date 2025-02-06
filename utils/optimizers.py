"""
This module provides some optimizers like gradient descend, ADAM.
"""

import logging
import numpy as np


class optimizer():

    def __init__(self, optimizer_hyper):
        self.method = optimizer_hyper.pop("method", "ADAM")

        upper_method = self.method.upper()
        if upper_method == "DEMON":
            self.optimizer = demon_adam_optim(**optimizer_hyper)
        elif upper_method in ("DECAY", "DECAY-ADAM"):
            self.optimizer = decay_adam(**optimizer_hyper)
        elif upper_method in ("GD", "GRADIENT-DESCENT", "GRAD-DESC"):
            self.optimizer = gradient_descent_optim(**optimizer_hyper)
        elif upper_method in ("SGD", "STOCHASTIC-GRADIENT-DESCENT", "STOCHASTIC-GRAD-DESC"):
            self.optimizer = stochastic_gradient_descent_optim(**optimizer_hyper)
        elif upper_method in ("GD2", "GRADIENT-DESCENT-2", "GRAD-DESC-2"):
            self.optimizer = gradient_descent_optim_2(**optimizer_hyper)
        elif upper_method in ("ADAM"):
            self.optimizer = adam_optim(**optimizer_hyper)
        else:
            raise ValueError(f"Optimizer method '{self.method}' not known.")

        self.hyper = self.optimizer.hyper
        # logging.info("used parameters: %s", self.hyper)

    def update(self, parameter_values, gradient_values):
        return self.optimizer.update(parameter_values, gradient_values)


class adam_optim():
    # from I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning (The MIT
    # Press, Cambridge, Massachusetts, 2016).

    def __init__(self, num_parameters, eta=0.01, beta1=0.9, beta2=0.999, delta=1e-8, **kwargs):

        self.hyper = {
            "optimizer": "ADAM",
            "eta": eta,
            "beta1": beta1,
            "beta2": beta2,
            "delta": delta,
            "num_parameters": num_parameters,
            "unused": kwargs}

        self.time_step = 0

        self.s = np.zeros(self.hyper["num_parameters"])  # 1st order moment
        self.r = np.zeros(self.hyper["num_parameters"])  # 2nd order moment

    def update(self, parameter_values, gradient_values):
        self.time_step += 1

        self.s = self.hyper["beta1"] * self.s + (1 - self.hyper["beta1"]) * gradient_values
        self.r = self.hyper["beta2"] * self.r + (1 - self.hyper["beta2"]) * gradient_values**2

        s_hat = self.s / (1 - self.hyper["beta1"]**self.time_step)
        r_hat = self.r / (1 - self.hyper["beta2"]**self.time_step)

        update = -self.hyper["eta"] * s_hat / (np.sqrt(r_hat) + self.hyper["delta"])

        return parameter_values + update


class demon_adam_optim():
    '''Decaying Momentum in Adam: https://arxiv.org/pdf/1910.04952v3.pdf'''

    def __init__(self, num_parameters, optim_T, eta=0.001, beta1=0.9, beta2=0.999, delta=1e-8, **kwargs):

        self.hyper = {
            "optimizer": "DEMON",
            "optim_T": optim_T,
            "eta": eta,
            "beta1": beta1,
            "beta2": beta2,
            "delta": delta,
            "num_parameters": num_parameters,
            "unused": kwargs}

        self.time_step = 0

        self.s = np.zeros(self.hyper["num_parameters"])  # 1st order moment
        self.r = np.zeros(self.hyper["num_parameters"])  # 2nd order moment

    def update(
        self,
        parameter_values,
        gradient_values,
    ):
        self.time_step += 1

        p_t = 1 - self.time_step / self.hyper["optim_T"]
        beta_1 = self.hyper["beta1"] * (p_t / (1 - self.hyper["beta1"] + self.hyper["beta1"] * p_t))

        self.s = beta_1 * self.s + (1 - beta_1) * gradient_values
        self.r = self.hyper["beta2"] * self.r + (1 - self.hyper["beta2"]) * gradient_values**2

        s_hat = self.s / (1 - beta_1**self.time_step)
        r_hat = self.r / (1 - self.hyper["beta2"]**self.time_step)

        update = -self.hyper["eta"] * s_hat / (np.sqrt(r_hat) + self.hyper["delta"])

        return parameter_values + update


class decay_adam():

    def __init__(self,
                 num_parameters,
                 optim_T,
                 eta_init=0.5,
                 eta_final=0.01,
                 beta1=0.9,
                 beta2=0.999,
                 delta=1e-8,
                 **kwargs):

        self.hyper = {
            "optimizer": "DECAY",
            "optim_T": optim_T,
            "eta_init": eta_init,
            "eta_final": eta_final,
            "beta1": beta1,
            "beta2": beta2,
            "delta": delta,
            "num_parameters": num_parameters,
            "unused": kwargs}

        self.hyper["decay_rate"] = np.log(eta_init / eta_final) / optim_T

        self.time_step = 0
        self.eta = eta_init  # just init

        self.s = np.zeros(num_parameters)  # 1st order moment
        self.r = np.zeros(num_parameters)  # 2nd order moment

    def update(self, parameter_values, gradient_values):
        self.time_step += 1
        self.eta = self.hyper["eta_init"] * np.exp(-self.hyper["decay_rate"] * self.time_step)

        self.s = self.hyper["beta1"] * self.s + (1 - self.hyper["beta1"]) * gradient_values
        self.r = self.hyper["beta2"] * self.r + (1 - self.hyper["beta2"]) * gradient_values**2

        s_hat = self.s / (1 - self.hyper["beta1"]**self.time_step)
        r_hat = self.r / (1 - self.hyper["beta2"]**self.time_step)

        update = -self.eta * s_hat / (np.sqrt(r_hat) + self.hyper["delta"])

        return parameter_values + update


class gradient_descent_optim_2():

    def __init__(self, optim_T_1, eta_1=0.1, eta_2=0.01, **kwargs):

        self.hyper = {"optimizer": "GD", "optim_T_1": optim_T_1, "eta_1": eta_1, "eta_2": eta_2, "unused": kwargs}
        self.time_step = 0

    def update(self, parameter_values, gradient_values):
        self.time_step += 1

        if self.time_step < self.hyper["optim_T_1"]:
            eta = self.hyper["eta_1"]
        else:
            eta = self.hyper["eta_2"]

        parameter_values -= eta * gradient_values

        return parameter_values


class gradient_descent_optim():

    def __init__(self, eta=0.01, **kwargs):

        self.hyper = {"optimizer": "GD", "eta": eta, "unused": kwargs}

    def update(self, parameter_values, gradient_values):

        parameter_values -= self.hyper["eta"] * gradient_values

        return parameter_values
    
class stochastic_gradient_descent_optim():

    def __init__(self, eta=0.01, **kwargs):

        self.hyper = {"optimizer": "SGD", "eta": eta, "unused": kwargs}

    def update(self, parameter_values, gradient_values):

        parameter_values -= self.hyper["eta"] * gradient_values * np.random.normal(0, self.hyper["unused"]['std'], size = gradient_values.shape)

        return parameter_values


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parameters = {"method": "ADAM", "num_parameters": 10, "eta": 0.1, "beta1": 0.9, "beta2": 0.99, "eta_final": 0.001}
    # opti = adam_optim(**parameters)
    ADAM = optimizer(parameters)

    vals = np.arange(10)
    for _ in range(100):
        vals = ADAM.update(vals, vals**2)
    print(np.round(vals, 2))

    DEMON = optimizer(dict(method="DEMON", num_parameters=10, optim_T=100, eta=0.1, beta1=0.9, beta2=0.99))
    vals = np.arange(10)
    for _ in range(100):
        vals = DEMON.update(vals, vals**2)
    print(np.round(vals, 2))

    DECAY = optimizer(
        dict(method="DECAY", num_parameters=10, optim_T=100, eta_init=0.5, eta_final=0.001, beta1=0.9, beta2=0.99))
    vals = np.arange(10)
    for _ in range(100):
        vals = DECAY.update(vals, vals**2)
        # print(DECAY.optimizer.eta)
    print(np.round(vals, 2))

    GD = optimizer(dict(method="GD", num_parameters=10, T=100, eta=0.5, beta1=0.9, beta2=0.99))
    vals = np.arange(10, dtype=float) / 10
    for _ in range(10):
        vals = GD.update(vals, vals**2)
    print(np.round(vals, 2))
