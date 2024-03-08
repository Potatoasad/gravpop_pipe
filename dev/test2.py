from gravpop_pipe import *
from gravpop import *
import jax
import jax.numpy as jnp

# Example usage:
filename = '/Users/asadh/Documents/Data/event_data2.h5'
selection_filename = '/Users/asadh/Documents/Data/selection_function2.h5'
data = load_hdf5_to_jax_dict(filename)

SM = SmoothedTwoComponentPrimaryMassRatio(primary_mass_name="mass_1_source")
R = PowerLawRedshift()

I = InferenceStandard.from_file(
					event_data_filename = filename,
					selection_data_filename = selection_filename,
					models = [SM,R]
					)

HL = PopulationLikelihood(
						 models  = [SM, R],
					     event_data = I.event_data,
					     selection_data = I.selection_data
					     )

Lambda_0 = dict(
	alpha = 3.5,
    lam = 0.04,
    mmin = 5,
    mmax = 96,
    beta = 1.1,
    mpp = 35,
    sigpp = 4,
    delta_m = 3,
#    mu_1 = 0.4,
#    sigma_1 = 0.1,
#    mu_2 = 0.2,
#    sigma_2 = 0.1,
    lamb = 2.9
)


import numpyro.distributions as dist
priors = dict(
	alpha 	= dist.Uniform(3,4),
    lam 	= dist.Uniform(0,1),
    mmin 	= dist.Uniform(5,6),
    mmax 	= dist.Uniform(95,100),
    beta 	= dist.Uniform(0,2),
    mpp 	= dist.Uniform(30,40),
    sigpp 	= dist.Uniform(3,5),
    delta_m = dist.Uniform(0,1),
#    mu_1 	= dist.Uniform(0,1),
#    sigma_1 = dist.Uniform(0,3),
#    mu_2 	= dist.Uniform(0,1),
#    sigma_2 = dist.Uniform(0,3),
    lamb 	= dist.Uniform(2,4)
)

latex_symbols = dict(
	alpha 	= r"$\alpha$",
    lam 	= r"$\lambda$",
    mmin 	= r"$m_{min}$",
    mmax 	= r"$m_{max}$",
    beta 	= r"$\beta$",
    mpp 	= r"$\mu_{m}$",
    sigpp 	= r"$\sigma_{m}$",
    delta_m = r"$\delta_m$",
#    mu_1 	= r"$\mu_1$",
#    sigma_1 = r"$\sigma_1$",
#    mu_2 	= r"$\mu_2$",
#    sigma_2 = r"$\sigma_2$",
    lamb 	= r"$\kappa$"
)

sampler = Sampler(
    priors = priors,
    latex_symbols = latex_symbols,
    likelihood = HL,
    num_samples = 200,
    num_warmup = 200,
    target_accept_prob = 0.7
)

print(HL.logpdf(Lambda_0))

import numpyro
numpyro.validation_enabled(True)
numpyro.enable_validation(True)

sampler.sample()