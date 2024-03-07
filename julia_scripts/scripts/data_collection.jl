using Pkg
env_dir = join(split(@__FILE__, "/")[1:(end-2)], "/")
Pkg.activate(env_dir)

using Revise, DataFrames
using TruncatedGaussianMixtures
using RingDB
using julia_scripts
using LaTeXStrings


database_folder = joinpath(homedir(), "Documents/Data/ringdb")
db = Database(database_folder)
event_list = readlines(joinpath(homedir(), "Documents/Data/selected_events_old.txt"))

### Some settings
N_samples_to_fit_with = 8000
N_samples_per_kernel = 1000
variables = [:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift]
a = [2.0, 0.0, 0.0, 0.0, 0.0]
b = [200.0, 1.0, 1.0, 1.0, 3.0]

settings = Dict(
				:N_samples_to_fit_with => 8000,
				:N_samples_per_kernel => 1000,
				:N_components => 5,
				:variables => [:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift],
				:sampling_variables => [:mass_1_source, :mass_ratio, :redshift],
				:analytical_variables => [:chi_1, :chi_2],
				:a => [2.0, 0.0, 0.0, 0.0, 0.0],
				:b => [200.0, 1.0, 1.0, 1.0, 3.0]
)

z_max = 3.0
priors = [
		EuclidianDistancePrior(:redshift, z_max=z_max),
		DetectorFrameMassesPrior(),
		FromSecondaryToMassRatio([:mass_1_source])
	]

prior = ProductPrior(priors)

β_schedule = AnnealingSchedule(;β_max=1.5, dβ_rise=0.01, dβ_relax=0.01, N_high=20, N_post=200)

transformation = Transformation(
	[:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift],
	(m1,q,χ1,χ2,z) -> (m1*to_chirp_mass(q)*(1+z), q,χ1,χ2,z),
	[:chirp_mass_det, :mass_ratio, :chi_1, :chi_2, :redshift],
	(M1,q,χ1,χ2,z) -> (M1/(to_chirp_mass(q)*(1+z)), q,χ1,χ2,z),
	[:prior]
)

event_dictionaries = []

using ProgressMeter

@showprogress for event_name ∈ event_list
	event = Event(db, event_name)
	the_dict = grab_event_data(event, prior, transformation, β_schedule, settings)
	push!(event_dictionaries, the_dict)
end


save_list_of_dicts_to_hdf5(event_dictionaries, event_list, joinpath(homedir(), "Documents/Data/event_data.h5"))





