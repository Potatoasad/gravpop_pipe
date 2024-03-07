from gravpop import *
from gravpop import AbstractPopulationModel
from .reading_data import *

from typing import List, Union, Dict
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class Inference:
	variables : List[str]
	sampled_models : List[AbstractPopulationModel]
	analytic_model : List[AbstractPopulationModel]
	event_data: Dict[str, jax.Array]
	selection_data: Dict[str, jax.Array]

	## Save data as an HDF5 with variable specifications, transformations and etc on both selection and events
	## Use data for inference with this class
	
	@classmethod
	def from_file(cls, variables, event_data_filename, selection_data_filename, sampled_models, analytic_models):
		event_data = stack_nested_jax_arrays(load_hdf5_to_jax_dict(event_data_filename))
		selection_data = load_hdf5_to_jax_dict(selection_data_filename)
		selection_data = selection_data["selection"]
		return cls(variables, sampled_models, analytic_models, event_data, selection_data)