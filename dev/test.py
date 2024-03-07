from gravpop_pipe import *
from gravpop import *
import jax
import jax.numpy as jnp

# Example usage:
filename = '/Users/asadh/Documents/Data/event_data.h5'
selection_filename = '/Users/asadh/Documents/Data/selection_function.h5'
data = load_hdf5_to_jax_dict(filename)

SM = SmoothedTwoComponentPrimaryMassRatio(primary_mass_name="mass_1_source")
R = PowerLawRedshift()
S1 =TruncatedGaussian1DAnalytic(a=0, b=1, var_name='chi_1', mu_name='mu_1', sigma_name='sigma_1')
S2 =TruncatedGaussian1DAnalytic(a=0, b=1, var_name='chi_2', mu_name='mu_2', sigma_name='sigma_2')

I = Inference.from_file(
					variables = ["mass_1_source", "mass_ratio", "chi_1", "chi_2", "redshift"],
					event_data_filename = filename,
					selection_data_filename = selection_filename,
					sampled_models = [SM,R],
					analytic_models = [S1,S2]
					)

HL = HybridPopulationLikelihood(
								 sampled_models  = [SM, R],
							     analytic_models =  [S1, S2],
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
    mu_1 = 0.4,
    sigma_1 = 0.1,
    mu_2 = 0.2,
    sigma_2 = 0.1,
    lamb = 2.9
)


import numpyro.distributions as dist
priors = dict(
	alpha 	= dist.Uniform(-1,6),
    lam 	= dist.Uniform( 0,1),
    mmin 	= dist.Uniform( 2,20),
    mmax 	= dist.Uniform(80,100),
    beta 	= dist.Uniform(-1,5),
    mpp 	= dist.Uniform(10,80),
    sigpp 	= dist.Uniform(2,10),
    delta_m = dist.Uniform(0,5),
    mu_1 	= dist.Uniform(0,1),
    sigma_1 = dist.Uniform(0,3),
    mu_2 	= dist.Uniform(0,1),
    sigma_2 = dist.Uniform(0,3),
    lamb 	= dist.Uniform(-8,8)
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
    mu_1 	= r"$\mu_1$",
    sigma_1 = r"$\sigma_1$",
    mu_2 	= r"$\mu_2$",
    sigma_2 = r"$\sigma_2$",
    lamb 	= r"$\kappa$"
)


print(HL.logpdf(Lambda_0))

sampler = Sampler(
    priors = priors,
    latex_symbols = latex_symbols,
    likelihood = HL,
    num_samples = 20,
    num_warmup = 200,
    target_accept_prob = 0.6
)


sampler.sample()
