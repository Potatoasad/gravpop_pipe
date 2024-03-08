using Pkg
env_dir = join(split(@__FILE__, "/")[1:(end-2)], "/")
Pkg.activate(env_dir)

using Revise, GLMakie, PairPlots, DataFrames, TruncatedGaussianMixtures, LaTeXStrings
using RingDB
using julia_scripts


## Set Parameters
path = "/Users/asadh/Downloads/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
columns = [:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift, :prior]

## Get Data
catalog = O1_O2_O3_sensitivity(path)
df_selections = get_injections(catalog)

N_samples_to_fit_with = 30_000

## Implement Cuts
df_detected = implement_cuts(df_selections; ifar_threshold = 1, snr_threshold = 11, m_min = 2.0, m_max=100.0, z_max = 3.0)
rename!(df_detected, :mass1_source  => :mass_1_source)
rename!(df_detected, :sampling_pdf  => :prior)
selection_samples = sample(df_detected, N_samples_to_fit_with, columns)

selection_dict = Dict()
for column ∈ columns
	selection_dict[column] = df[!, column]
end

using HDF5

function save_dict_to_file(filename, the_dictionary; group_name="selection")
	h5open(filename, "w") do file
        group = create_group(file, group_name)
        save_dict_to_group(group, the_dictionary)
    end
end

save_dict_to_file(joinpath(homedir(), "Documents/Data/selection_function2.h5"), selection_dict)







database_folder = joinpath(homedir(), "Documents/Data/ringdb")
db = Database(database_folder)
event_list = readlines(joinpath(homedir(), "Documents/Data/selected_events_old.txt"))

z_max = 3.0
priors = [
		EuclidianDistancePrior(:redshift, z_max=z_max),
		DetectorFrameMassesPrior(),
		FromSecondaryToMassRatio([:mass_1_source])
	]

prior = ProductPrior(priors)

N_samples_to_fit_with = 1_000

using ProgressMeter

event_dictionaries = []

@showprogress for event_name ∈ event_list
	event = Event(db, event_name)
	post = event.posteriors()
	select!(post, :, [:spin_1x, :spin_1y, :spin_1z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_1)
	select!(post, :, [:spin_2x, :spin_2y, :spin_2z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_2)

	post = sample(post, N_samples_to_fit_with, variables)

	### Evaluate Prior on event posterior samples
	evaluate!(post, prior, :prior)

	the_dict = Dict()
	for col ∈ columns
		the_dict[col] = post[!, col]
	end
	push!(event_dictionaries, the_dict)
end


save_list_of_dicts_to_hdf5(event_dictionaries, event_list, joinpath(homedir(), "Documents/Data/event_data2.h5"))


