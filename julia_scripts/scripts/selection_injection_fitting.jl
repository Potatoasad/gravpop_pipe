using Pkg
env_dir = join(split(@__FILE__, "/")[1:(end-2)], "/")
Pkg.activate(env_dir)

using Revise, GLMakie, PairPlots, DataFrames, TruncatedGaussianMixtures, LaTeXStrings
using RingDB
using julia_scripts


## Set Parameters
path = "/Users/asadh/Downloads/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
columns = [:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift, :prior]

N_samples_to_fit_with = 30_000
N_components = 10
N_samples_per_kernel = 4_000
a = [2.0, 0.0, 0.0, 0.0, 0.0]
b = [100.0, 1.0, 1.0, 1.0, 3.0]

## Get Data
catalog = O1_O2_O3_sensitivity(path)
df_selections = get_injections(catalog)
analysis_time = get_analysis_time(catalog)
total_generated = get_total_generated(catalog)

## Implement Cuts
df_detected = implement_cuts(df_selections; ifar_threshold = 1, snr_threshold = 11, m_min = 2.0, m_max=100.0, z_max = 3.0)
rename!(df_detected, :mass1_source  => :mass_1_source)
#rename!(df_detected, :sampling_pdf  => :prior)
select!(df_detected, :, [:sampling_pdf, :mass_1_source] => ByRow((p,m1) -> p*m1) => :prior)
select!(df_detected, :, [:prior, :chi_1] => ByRow((p,χ₁) -> p*2π*χ₁^2) => :prior)
select!(df_detected, :, [:prior, :chi_2] => ByRow((p,χ₂) -> p*2π*χ₂^2) => :prior)
selection_samples = sample(df_detected, N_samples_to_fit_with, columns)

## Set up fitting procedure
β_schedule = AnnealingSchedule(;β_max=2.0, dβ_rise=0.01, dβ_relax=0.01, N_high=100, N_post=200)

identity_transformation = RingDB.Transformation(
	[:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift],
	(m1,q,χ1,χ2,z) -> (m1,q,χ1,χ2,z),
	[:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift],
	(M1,q,χ1,χ2,z) -> (M1,q,χ1,χ2,z),
	[:prior]
)


mix, df = fit_gmm(selection_samples, N_components, a, b, identity_transformation, β_schedule; progress=true, cov=:diag)

compare_distributions(df[!, [:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift]], mix; columns=[L"M_{chirp}^{det}", L"q", L"\chi_1", L"\chi_2", L"z"])

sampling_variables = [:mass_1_source, :mass_ratio, :redshift]
analytical_variables = [:chi_1, :chi_2]
variables = [:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift]
the_dictionary = post_process_event_fit(df, mix, sampling_variables, analytical_variables, variables; N_samples_per_kernel=N_samples_per_kernel)


using HDF5

#function save_dict_to_file(filename, the_dictionary; group_name="selection")
#	h5open(filename, "w") do file
#        group = create_group(file, group_name)
#        save_dict_to_group(group, the_dictionary)
#    end
#end


function save_dict_to_file(filename, the_dictionary; group_name="selection")
	h5open(filename, "w") do file
        group = create_group(file, group_name)
        save_dict_to_group(group, the_dictionary)
        write_attribute(group, "analysis_time", analysis_time)
        write_attribute(group, "total_generated", total_generated)
    end
end

save_dict_to_file(joinpath(homedir(), "Documents/Data/selection_function_prior_fix.h5"), the_dictionary)


