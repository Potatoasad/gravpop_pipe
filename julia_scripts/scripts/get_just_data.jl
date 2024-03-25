using Pkg
env_dir = join(split(@__FILE__, "/")[1:(end-2)], "/")
Pkg.activate(env_dir)

using Revise, GLMakie, PairPlots, DataFrames, TruncatedGaussianMixtures, LaTeXStrings
using RingDB
using julia_scripts


## Set Parameters
path = "/Users/asadh/Downloads/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
columns = [:mass_1_source, :mass_ratio, :chi_1, :chi_2, :cos_tilt_1, :cos_tilt_2, :redshift, :prior]

## Get Data
catalog = O1_O2_O3_sensitivity(path)
df_selections = get_injections(catalog)
analysis_time = get_analysis_time(catalog)
total_generated = get_total_generated(catalog)

N_samples_to_fit_with = 30_000

## Implement Cuts
df_detected = implement_cuts(df_selections; ifar_threshold = 1, snr_threshold = 11, m_min = 2.0, m_max=100.0, z_max = 1.9)
rename!(df_detected, :mass1_source  => :mass_1_source)
#rename!(df_detected, :sampling_pdf  => :prior) <-this  is wrong!!
select!(df_detected, :, [:sampling_pdf, :mass_1_source] => ByRow((p,m1) -> p*m1) => :prior)
#select!(df_detected, :, [:prior, :chi_1] => ByRow((p,χ₁) -> p*2π*χ₁^2) => :prior)
#select!(df_detected, :, [:prior, :chi_2] => ByRow((p,χ₂) -> p*2π*χ₂^2) => :prior)
selection_samples = sample(df_detected, N_samples_to_fit_with, columns)

selection_dict = Dict()
for column ∈ columns
	selection_dict[column] = selection_samples[!, column]
end

using HDF5

function save_dict_to_file(filename, the_dictionary; group_name="selection")
	h5open(filename, "w") do file
        group = create_group(file, group_name)
        save_dict_to_group(group, the_dictionary)
        write_attribute(group, "analysis_time", analysis_time)
        write_attribute(group, "total_generated", total_generated)
    end
end

save_dict_to_file(joinpath(homedir(), "Documents/Data/selection_function_fixed_z_max_1p9_no_spin_jacobian.h5"), selection_dict)






"""
#database_folder = joinpath(homedir(), "Documents/Data/ringdb")
#db = Database(database_folder)
#event_list = readlines(joinpath(homedir(), "Documents/Data/selected_events_old.txt"))

#database_folder = joinpath(homedir(), "Documents/Data/ringdb")
#db = Database(database_folder)
#event_list = readlines(joinpath(homedir(), "Documents/Data/posterior_names.txt"))
db = GWPopPosteriorFile("/Users/asadh/Documents/Data/posteriors.pkl","/Users/asadh/Documents/Data/posterior_names.txt")

z_max = 1.9
priors = [
		EuclidianDistancePrior(:redshift, z_max=z_max),
		DetectorFrameMassesPrior(),
		FromSecondaryToMassRatio([:mass_1_source])
	]

prior = ProductPrior(priors)

N_samples_to_fit_with = 1_000

using ProgressMeter

event_dictionaries = []
event_list = String[]
@showprogress for (event_name, post) ∈ db.posteriors
	#event = Event(db, event_name)
	#post = event.posteriors()
	#select!(post, :, :mass_1 => ByRow(x -> x) => :mass_1_source)
	rename!(post, :mass_1 => :mass_1_source)
	rename!(post, :a_1 => :chi_1)
	rename!(post, :a_2 => :chi_2)
	#select!(post, :, [:spin_1x, :spin_1y, :spin_1z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_1)
	#select!(post, :, [:spin_2x, :spin_2y, :spin_2z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_2)


	post = sample(post, N_samples_to_fit_with, columns)

	### Evaluate Prior on event posterior samples
	evaluate!(post, prior, :prior_ringdb)

	the_dict = Dict()
	for col ∈ columns
		the_dict[col] = post[!, col]
	end
	the_dict[:prior_ringdb] = post[!, :prior_ringdb]
	push!(event_dictionaries, the_dict)
	push!(event_list, event_name)
end


save_list_of_dicts_to_hdf5(event_dictionaries, event_list, joinpath(homedir(), "Documents/Data/event_data_from_pickle.h5"))

"""
