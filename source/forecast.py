import pymc as pm
import numpy as np
from source.model_output import SEIRModelOutput, SEIRParams
from source.SEIR_network import SEIRNetworkModel


class SEIRForecaster():
    def __init__(self, initial_params: SEIRParams):
        self.best_params = initial_params

    def calibrate(self, model, observed):
        epsilon = 2000
        sample = 100

        print(np.where(observed == 0))
        epidemic_len = len(observed)
        zeros_arr = np.where(observed == 0)[0]
        if len(zeros_arr) > 1:
            epidemic_len = zeros_arr[1]

        def simulation_func(rng, alpha, beta, gamma,
                            delta, init_inf_frac, size=None):
            return model.simulate(alpha, beta, gamma, delta, init_inf_frac)[:epidemic_len]

        with pm.Model() as PMmodel:
            alpha = pm.Uniform(
                name="alpha", lower=.001, upper=1)
            beta = pm.Uniform(
                name="beta", lower=0.001, upper=1)

            sim = pm.Simulator(
                'sim',
                simulation_func,
                params=(alpha, beta, self.best_params.gamma,
                        self.best_params.delta, self.best_params.init_inf_frac),
                epsilon=epsilon,
                observed=observed[:epidemic_len],
            )
            idata = pm.sample_smc(progressbar=True, draws=5000)

        posterior = idata.posterior.stack(samples=("draw", "chain"))
        alpha_arr = posterior["alpha"].values
        beta_arr = posterior["beta"].values
        self.best_params.alpha = alpha_arr.mean()
        self.best_params.beta = beta_arr.mean()
        return {'alpha_arr': alpha_arr, 'beta_arr': beta_arr}
