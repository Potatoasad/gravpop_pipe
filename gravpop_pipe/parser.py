import re
import configparser
from gravpop import *
import pandas as pd

def config_to_dict(config):
	config_dict = {}
	for section in config.sections():
		parts = section.split('.')
		current_dict = config_dict
		for part in parts[:-1]:
			current_dict = current_dict.setdefault(part, {})
		current_dict[parts[-1]] = dict(config.items(section))
	return config_dict

convert_to_list = lambda x: eval(x.replace(', ', "','").replace("[", "['").replace("]", "']"))

def has_empty_function_call(string):
	pattern = r'[^(]*\(\s*\)'
	return re.search(pattern, string) is not None

class Parser:
	def __init__(self, config_file):
		self.config_file = config_file
		self.config = configparser.ConfigParser()
		self.config.read(self.config_file)
		self.config_dict = config_to_dict(self.config)
		self.default_model_order = ['mass', 'redshift', 'spin_magnitude', 'spin_orientation']
		self.default_model_order = [m for m in self.default_model_order if m in self.config_dict["Models"].keys()]
		self._likelihood = None
		self._priors = None
		self._latex_names = None
		self._sampler = None
		self._hyper_posterior = None
		params = [convert_to_list(self.config_dict['Variables']['Population'][m]) for m in self.default_model_order if m in self.default_model_order]
		hyper_params = []
		for p in params:
		    hyper_params += p
		self.hyper_parameters = hyper_params
		
	def get_list_of_models(self, model_dict):
		models = self.models
		return [models[m] for m in self.default_model_order if m in models] + list([v for m,v in models.items() if m not in self.default_model_order])
		

	@property
	def save_locations(self):
		return self.config_dict.get('Output', {})

	@property
	def models(self):
		new_names = {}
		for model in self.config_dict['Models'].keys():
			hyper_var_names = convert_to_list(self.config_dict['Variables']['Population'][model])
			var_names = convert_to_list(self.config_dict['Variables']['Event'][model])
			model_call = self.config_dict['Models'][model]
			if has_empty_function_call(model_call):
				new_names[model] = model_call.replace(')',  f'var_names={var_names}, hyper_var_names={hyper_var_names})')
			else:
				new_names[model] = model_call.replace(')',  f', var_names={var_names}, hyper_var_names={hyper_var_names})')
				
		return {key: eval(value) for key, value in new_names.items()}
	
	@property
	def likelihood(self):
		if self._likelihood is None:
			likelihood_items = {key:eval(value) for key, value in self.config_dict['Likelihood'].items()}
			cls = likelihood_items.pop('type', PopulationLikelihood)
			self._likelihood = cls.from_file(event_data_filename=self.config_dict['DataSources']['event_data'],
											 selection_data_filename=self.config_dict['DataSources']['selection_data'], 
											 models=self.get_list_of_models(self.models),
											 **likelihood_items)
			
		return self._likelihood
	
	@property
	def priors(self):
		if self._priors is None:
			import numpyro.distributions as dist
			self._priors = {parameter : eval(prior_string) for parameter, prior_string in self.config_dict['Priors'].items()}
		
		return self._priors
	
	@property
	def latex_names(self):
		if self._latex_names is None:
			self._latex_names = {key: eval(value) for key,value in self.config_dict['Latex'].items()}
		
		return self._latex_names

	@property
	def constraints(self):
		if 'Constraints' in self.config_dict:
			return [eval(a) for a in self.config_dict['Constraints'].values()]
	
	@property
	def sampler(self):
		if self._sampler is None:
			kwargs = {key:eval(value) for key, value in self.config_dict['Sampler'].items()}
			self._sampler = Sampler(priors = self.priors,
									constraints = self.constraints,
									latex_symbols = self.latex_names,
									likelihood = self.likelihood,
									**kwargs)
			
		return self._sampler

	def load_lvk(self, samples_location, n_samples=8000):
		samples_location = samples_location or self.save_locations.get('samples', None)
		from bilby.core.result import read_in_result
		PP_path = samples_location
		PP_result = read_in_result(PP_path)
		PP_hyperposterior_samples = PP_result.posterior.sample(n_samples).copy()
		cols = self.models
		self.sampler.samples = PP_hyperposterior_samples[self.hyper_parameters]
	
	@property
	def hyper_posterior(self):
		if self.sampler.samples is None:
			raise ValueError("Need to run the sampler first")
			
		if self._hyper_posterior is None:
			self._hyper_posterior = HyperPosterior(self.sampler.samples, self.likelihood, models=self.models)
			
		return self._hyper_posterior

	def run(self, samples_save_location=None):
		samples_save_location = samples_save_location or self.save_locations.get('samples', None)
		self.sampler.sample()
		self.sampler.samples.to_csv(samples_save_location, index=False)
		print(f"Samples saved at {samples_save_location}")

	def load(self, samples_location=None):
		samples_location = samples_location or self.save_locations.get('samples', None)
		self.sampler.samples = pd.read_csv(samples_location)[self.hyper_parameters]


		
