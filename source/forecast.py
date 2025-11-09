import pymc as pm
import numpy as np
import arviz as az
from source.model_output import SEIRModelOutput, SEIRParams
from source.SEIR_network import SEIRNetworkModel


class SEIRForecaster():
    def __init__(self, initial_params: SEIRParams):
        self.best_params = initial_params

    def calibrate(self, model, full_observed, period, epsilon=2000):
        observed = full_observed[:period].copy()
        
        '''
        zeros_arr = np.where(observed == 0)[0]
        if len(zeros_arr) > 1:
            epidemic_len = zeros_arr[1]
        '''
        def simulation_func(rng, alpha, beta, gamma,
                            delta, init_inf_frac, epidemic_len, size=None):
            epidemic_len = np.array(epidemic_len).flatten()[0]
            return model.simulate(alpha, beta, gamma, delta, init_inf_frac)[:epidemic_len]

        with pm.Model() as PMmodel:
            real_data = pm.Data("incidence", 
                        observed)
            alpha = pm.Uniform(
                name="alpha", lower=.0001, upper=1)
            beta = pm.Uniform(
                name="tau", lower=0.0, upper=1)
            #epidemic_len = len(observed)
            sim = pm.Simulator(
                'sim',
                simulation_func,
                params=(alpha, beta, self.best_params.gamma,
                        self.best_params.delta, self.best_params.init_inf_frac,
                       real_data.shape[0]),
                epsilon=epsilon,
                observed=real_data,
            )
            idata = pm.sample_smc(progressbar=True, draws=2000, chains=4)
            
            if len(full_observed)==period:
                print('full calibr')
                idata.extend(pm.sample_posterior_predictive(idata))
            
        if len(full_observed)!=period:
            with PMmodel:
                #out-onew_observedle
                pm.set_data({'incidence':
                              full_observed})
                #epidemic_len = len(full_observed)
                idata = pm.sample_posterior_predictive(
                        idata, 
                        var_names=["sim"],
                        extend_inferencedata=True, 
                        predictions=True)
    
        # az.plot_pair(
        #     idata,
        #     var_names=["beta", "alpha"],
        #     kind=["scatter", "kde"],
        #     kde_kwargs={"fill_last": False},
        #     marginals=True,
        #     point_estimate="mode",
        #     figsize=(11.5, 5),
        # )
        posterior = idata.posterior.stack(samples=("draw", "chain"))
        
        
        alpha_arr = posterior["alpha"].values
        beta_arr = posterior["tau"].values
        self.best_params.alpha = alpha_arr.mean()
        self.best_params.beta = beta_arr.mean()
        return idata
        # return {'alpha_arr': alpha_arr, 'beta_arr': beta_arr}
