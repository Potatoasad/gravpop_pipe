from gravpop_pipe import *
from gravpop import *
import jax
import jax.numpy as jnp

# Example usage:
filename = '/Users/asadh/Documents/Data/event_data_from_pickle.h5'
selection_filename = '/Users/asadh/Documents/Data/selection_function_fixed_z_max_1p9.h5'
data = load_hdf5_to_jax_dict(filename)

SM = SmoothedTwoComponentPrimaryMassRatio(primary_mass_name="mass_1_source")
R = PowerLawRedshift(z_max=1.9)

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
	alpha 	= dist.Uniform(-4,12),
    lam 	= dist.Uniform(0,1),
    mmin 	= dist.Uniform(2,10),
    mmax 	= dist.Uniform(70,100),
    beta 	= dist.Uniform(-2,7),
    mpp 	= dist.Uniform(20,50),
    sigpp 	= dist.Uniform(1,10),
    delta_m = dist.Uniform(0,12),
#    mu_1 	= dist.Uniform(0,1),
#    sigma_1 = dist.Uniform(0,3),
#    mu_2 	= dist.Uniform(0,1),
#    sigma_2 = dist.Uniform(0,3),
    lamb 	= dist.Uniform(-10,10)
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
    num_samples = 8000,
    num_warmup = 200,
    target_accept_prob = 0.6,
    just_prior = False
)

#print(HL.logpdf(Lambda_0))

#import numpyro
#numpyro.validation_enabled(True)
#numpyro.enable_validation(True)

#sampler.sample()

from jax import jacfwd
def model_gradient(model, data, param, canonical_parameter_order=None):
	canonical_parameter_order = canonical_parameter_order or list(param.keys())

	def make_vector(d):
		return jnp.array([d[param] for param in canonical_parameter_order])

	def make_dictionary(x):
		return {parameter : x[i] for i,parameter in enumerate(canonical_parameter_order)}

	dYdx = jacfwd(lambda x: model(data, make_dictionary(x)))(make_vector(param))
	if len(canonical_parameter_order) == 1:
		return {parameter : dYdx.flatten() for i,parameter in enumerate(canonical_parameter_order)}

	return {parameter : dYdx[..., i] for i,parameter in enumerate(canonical_parameter_order)}

def likelihood_gradient(likelihood, param, canonical_parameter_order=None):
	canonical_parameter_order = canonical_parameter_order or list(param.keys())

	def make_vector(d):
		return jnp.array([d[param] for param in canonical_parameter_order])

	def make_dictionary(x):
		return {parameter : x[i] for i,parameter in enumerate(canonical_parameter_order)}

	dYdx = jacfwd(lambda x: likelihood.logpdf(make_dictionary(x)))(make_vector(param))
	if len(canonical_parameter_order) == 1:
		return {parameter : dYdx.flatten() for i,parameter in enumerate(canonical_parameter_order)}

	return {parameter : dYdx[..., i] for i,parameter in enumerate(canonical_parameter_order)}

"""

Lambda_0 =  {'alpha': 3.5, 'beta': -1.1, 'mmin': 5, 'mmax': 90, 
             'mpp': 35, 'sigpp': 3, 'lam': 0.4, 'lamb':2.9, 'delta_m':3}

derivatives = model_gradient(SM, I.selection_data, Lambda_0)
print(f"For the point Λ₀ given by {Lambda_0}")
print("Mass models have no nans in their inferred gradients: ", not jnp.any(jnp.array([jnp.any(jnp.isnan(derivatives[x])) for x in derivatives.keys()])))

derivatives = likelihood_gradient(HL, Lambda_0)
print("likelihood has no nans in its inferred gradients: ", not jnp.any(jnp.array([jnp.any(jnp.isnan(derivatives[x])) for x in derivatives.keys()])))

"""




result = sampler.sample()
post = sampler.samples
post.to_csv(f"./samples_output.csv", index=False)
fig = sampler.corner()
fig.savefig("./test.png")
