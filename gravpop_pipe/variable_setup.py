from gravpop import *
from gravpop import AbstractPopulationModel, Redshift
from .reading_data import *

from typing import List, Union, Dict, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class Inference:
	sampled_models : List[AbstractPopulationModel]
	analytic_model : List[AbstractPopulationModel]
	event_data: Dict[str, jax.Array]
	selection_data: Dict[str, jax.Array]
	selection_attributes: Dict[str, Any]

	## Save data as an HDF5 with variable specifications, transformations and etc on both selection and events
	## Use data for inference with this class
	
	@classmethod
	def from_file(cls, event_data_filename, selection_data_filename, sampled_models, analytic_models):
		event_data = stack_nested_jax_arrays(load_hdf5_to_jax_dict(event_data_filename))
		selection_data = load_hdf5_to_jax_dict(selection_data_filename)
		selection_attributes = load_hdf5_attributes(selection_data_filename)
		if "selection" in selection_data.keys():
			selection_data = selection_data["selection"]

		if "selection" in selection_attributes.keys():
			selection_attributes = selection_attributes["selection"]

		return cls(sampled_models, analytic_models, event_data, selection_data, selection_attributes)


	def to_model(self, ModelClass=HybridPopulationLikelihood, SelectionClass=SelectionFunction):
		return ModelClass(self.sampled_models, self.analytic_models, self.event_data, self.selection_data)

@dataclass
class InferenceStandard:
	models : List[AbstractPopulationModel]
	event_data: Dict[str, jax.Array]
	selection_data: Dict[str, jax.Array]
	selection_attributes: Dict[str, Any]

	## Save data as an HDF5 with variable specifications, transformations and etc on both selection and events
	## Use data for inference with this class
	
	@classmethod
	def from_file(cls, event_data_filename, selection_data_filename, models):
		event_data = stack_nested_jax_arrays(load_hdf5_to_jax_dict(event_data_filename))
		selection_data = load_hdf5_to_jax_dict(selection_data_filename)
		selection_attributes = load_hdf5_attributes(selection_data_filename)
		if "selection" in selection_data.keys():
			selection_data = selection_data["selection"]

		if "selection" in selection_attributes.keys():
			selection_attributes = selection_attributes["selection"]
		return cls(models, event_data, selection_data, selection_attributes)

	def to_model(self, ModelClass=PopulationLikelihood, SelectionClass=SelectionFunction):
		self.redshift_model = [model for model in self.models if isinstance(model, Redshift)]
		if len(self.redshift_model) == 0:
			self.redshift_model = None
		else:
			self.redshift_model = self.redshift_model[0]
		self.selection = SelectionClass(self.selection_data, 
								   self.selection_attributes['analysis_time'],
								   self.selection_attributes['total_generated'],
								   self.redshift_model)
		return ModelClass(self.models, self.event_data, self.selection)






