import jax
import jax.numpy as jnp
from gravpop import *
from gravpop_pipe import *
import matplotlib.pyplot as plt

import gwpopulation
gwpopulation.disable_cupy()


# Example usage:
filename = '/Users/asadh/Documents/Data/event_data2.h5'
selection_filename = '/Users/asadh/Documents/Data/selection_function2.h5'

SM = SmoothedTwoComponentPrimaryMassRatio(primary_mass_name="mass_1_source")
R = PowerLawRedshift(z_max=3.0)

I = InferenceStandard.from_file(
					event_data_filename = filename,
					selection_data_filename = selection_filename,
					models = [SM,R]
					)