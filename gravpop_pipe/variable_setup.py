from gravpop import *
from gravpop import AbstractPopulationModel
from typing import List, Union, Dict
from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass
class Inference:
	variables : List[str]
	sampling_variables : List[str]
	analytic_variables : List[str]
	event_data: Dict[str, jax.Array]
	selection_data: Dict[str, jax.Array]
	models: List[AbstractPopulationModel] # Not coming from the data specification

	## Save data as an HDF5 with variable specifications, transformations and etc on both selection and events
	## Use data for inference with this class
	
