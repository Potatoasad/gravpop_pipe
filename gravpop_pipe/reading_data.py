import h5py
import jax.numpy as jnp

def load_hdf5_to_jax_dict(filename):
    data = {}
    with h5py.File(filename, 'r') as f:
        for event_name, event_group in f.items():
            event_data = {}
            for dataset_name, dataset in event_group.items():
                if isinstance(dataset, h5py.Dataset):
                    event_data[dataset_name] = jnp.array(dataset)
                elif isinstance(dataset, h5py.Group):
                    subgroup_data = {}
                    for subgroup_name, subgroup_dataset in dataset.items():
                        subgroup_data[subgroup_name] = jnp.array(subgroup_dataset)
                    event_data[dataset_name] = subgroup_data
            data[event_name] = event_data
    return data


def stack_nested_jax_arrays(data):
    stacked_data = {}
    for event_name, event_data in data.items():
        for var_name, var_data in event_data.items():
            if isinstance(var_data, dict):
                if var_name not in stacked_data:
                    stacked_data[var_name] = stack_nested_jax_arrays({event_name: var_data})
                else:
                    stacked_data[var_name].update(stack_nested_jax_arrays({event_name: var_data}))
            else:
                if var_name not in stacked_data:
                    stacked_data[var_name] = jnp.expand_dims(var_data, axis=0)
                else:
                    stacked_data[var_name] = jnp.vstack((stacked_data[var_name], jnp.expand_dims(var_data, axis=0)))
    return stacked_data