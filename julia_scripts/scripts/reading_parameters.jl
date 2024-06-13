using Pkg
env_dir = join(split(@__FILE__, "/")[1:(end-2)], "/")
Pkg.activate(env_dir)

#using Revise
using ConfParser
using ArgParse
using DataFrames, LaTeXStrings
using Makie, GLMakie, CairoMakie
using RingDB
using TruncatedGaussianMixtures
using HDF5

using julia_scripts

function nest_dictionary(dict::Dict{T, L}) where {T,L}
    nested_dict = Dict{Any, Any}()

    for (key, value) in dict
        parts = split(key, ".")
        current_dict = nested_dict

        for (i, part) in enumerate(parts)
            if i == length(parts)
                current_dict[part] = value
            else
                if !haskey(current_dict, part)
                    current_dict[part] = Dict{Any, Any}()
                elseif typeof(current_dict[part]) != Dict{Any, Any}
                    # If the value is not a dictionary, convert it to a dictionary
                    current_dict[part] = Dict(part => current_dict[part])
                end
                current_dict = current_dict[part]
            end
        end
    end

    return nested_dict
end

preprocess_var_name(x; var_map=Dict()) = (x ∈ keys(var_map)) ? var_map[x] : x
dict_eval(the_dictionary; var_map=Dict()) = Dict(preprocess_var_name(Meta.parse(key); var_map=var_map) => eval(Meta.parse(value[1])) for (key,value) ∈ the_dictionary if !(value isa Dict))

s = ArgParseSettings()
@add_arg_table s begin
    "config_filepath"
        help = "Path to the config (ini) file that contains the data collection settings"
        required = true
end

parsed_args = parse_args(ARGS, s)

filename = parsed_args["config_filepath"]

conf = ConfParse(filename)
parse_conf!(conf)


#### PARSE DATA SOURCE LOCATIONS ######################
event_name_list_location			 = conf._data["datasources"]["event_list"][1][2:(end-1)]
selection_function_samples_location  = conf._data["datasources"]["selection_function_samples"][1][2:(end-1)]
database_location 		    		 = conf._data["datasources"]["database_location"][1][2:(end-1)]
event_data_output_location			 = conf._data["datasources"]["event_data_output_location"][1][2:(end-1)]
selection_data_output_location		 = conf._data["datasources"]["selection_data_output_location"][1][2:(end-1)]


config_dict = nest_dictionary(conf._data)

#### SET UP DATABASE
db = Database(database_location)


###### EVENTS ################################
event_list = readlines(event_name_list_location)


### Sampling arguments
N_samples_to_fit_with = config_dict["events"]["sampling"]["N_samples_to_fit_with"][1] |> Meta.parse |> eval
N_samples_per_kernel = config_dict["events"]["sampling"]["N_samples_per_kernel"][1] |> Meta.parse |> eval
N_components = config_dict["events"]["sampling"]["N_components"][1] |> Meta.parse |> eval


### Variable sets
variable_names = (config_dict["events"]["variables"]["variable_names"] |> (x -> join(x, ",")) |> Meta.parse).args
variable_names = identity.(variable_names) ## Makes the type concrete

sampling_variable_names = (config_dict["events"]["variables"]["sampling_variables"] |> (x -> join(x, ",")) |> Meta.parse).args
sampling_variable_names = identity.(sampling_variable_names) ## Makes the type concrete

analytical_variable_names = (config_dict["events"]["variables"]["analytical_variables"] |> (x -> join(x, ",")) |> Meta.parse).args
analytical_variable_names = identity.(analytical_variable_names) ## Makes the type concrete

latex_variable_names = (config_dict["events"]["variables"]["latex_names"] |> (x -> join(x, ",")) |> Meta.parse).args
latex_variable_names = identity.(eval.(latex_variable_names)) ## Makes the type concrete


### Add in prior
priors = config_dict["events"]["prior"] |> values |> collect .|>  (x -> eval(Meta.parse(join(x, ","))))
prior = ProductPrior(priors)

### Variable Truncation Limits
limits = join(config_dict["events"]["variables"]["variable_truncation_limits"], ",") |> Meta.parse |> eval
limits = [Float64.(limit) for limit in limits]
model_a = [limit[1] for limit in limits]
model_b = [limit[2] for limit in limits]


### TGMM intermediate transformations
transformed_variable_names = (config_dict["events"]["prefittransformation"]["variable_names"] |> (x -> join(x, ",")) |> Meta.parse).args
transformed_variable_names = identity.(transformed_variable_names)

truncation_limits = join(config_dict["events"]["prefittransformation"]["variable_truncation_limits"], ",") |> Meta.parse |> eval
truncation_limits = [Float64.(limit) for limit in limits]
truncation_a = [limit[1] for limit in truncation_limits]
truncation_b = [limit[2] for limit in truncation_limits]

forward_transformation = config_dict["events"]["prefittransformation"]["transformation_function"] |> x->join(x,",") |> Meta.parse |> eval
inverse_transformation = config_dict["events"]["prefittransformation"]["inverse_transformation_function"] |> x->join(x,",") |> Meta.parse |> eval

transformation = TruncatedGaussianMixtures.Transformation(
	variable_names,
	forward_transformation,
	transformed_variable_names,
	inverse_transformation,
	[:prior]
)



### Extract the TGMM settings
settings = Dict(
				:N_samples_to_fit_with => N_samples_to_fit_with,
				:N_samples_per_kernel => N_samples_per_kernel,
				:N_components => N_components,
				:variables => variable_names,
				:sampling_variables => sampling_variable_names,
				:analytical_variables => analytical_variable_names,
				:a => truncation_a,
				:b => truncation_b
)

var_map = Dict( :beta_max => :β_max, :delta_beta_rise => :dβ_rise, :delta_beta_fall => :dβ_relax, :N_beta_max => :N_high)
preprocess_var_name(x) = (x ∈ keys(var_map)) ? var_map[x] : x
dict_eval(the_dictionary) = Dict(preprocess_var_name(Meta.parse(key)) => eval(Meta.parse(value[1])) for (key,value) ∈ the_dictionary if !(value isa Dict))

TGMM_kwargs = dict_eval(copy(conf._data["events.truncatedgaussianmixturemodelparameters"]))

TGMM_annealing_kwargs = dict_eval(copy(conf._data["events.truncatedgaussianmixturemodelparameters.annealingschedule"]))
β_schedule = AnnealingSchedule(;TGMM_annealing_kwargs...)



############# COMPUTE FOR ALL EVENTS ####################
event_dictionaries = []

using ProgressMeter

function get_samples(event, N_samples_to_fit_with, variables)
	post = event.posteriors()
	select!(post, :, [:spin_1x, :spin_1y, :spin_1z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_1)
	select!(post, :, [:spin_2x, :spin_2y, :spin_2z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_2)

	sample(post, N_samples_to_fit_with, variables)
end

final_df = vcat((@showprogress [get_samples(Event(db, event_name), N_samples_to_fit_with, variable_names) for event_name ∈ event_list])...)
@show describe(final_df)
final_df[!, :prior] .= 1.0
@show describe(forward(transformation, final_df))

@showprogress for event_name ∈ event_list
	event = Event(db, event_name)
	the_dict = grab_event_data(event, prior, transformation, β_schedule, settings)
	push!(event_dictionaries, the_dict)
end

#println(event_dictionaries)
#println(event_list[1:2])
#println(event_data_output_location)

save_list_of_dicts_to_hdf5(event_dictionaries, event_list, event_data_output_location)

############# DEBUG PLOTS ########################
if "event_data_plots" in keys(conf._data["datasources"])	
	event_data_plot_location = conf._data["datasources"]["event_data_plots"][1][2:(end-1)]

	using PDFmerger: append_pdf!
	CairoMakie.activate!()

	# delete current pdf if it doesn't exist
	rm(event_data_plot_location, force=true)

	@showprogress for i ∈ 1:length(event_dictionaries)
		df = event_dictionaries[i][:julia_data][:df]
		df2 = DataFrame(collect(rand(event_dictionaries[i][:julia_data][:gmm],nrow(df))'), transformed_variable_names)
		df2[!, :prior] .= 1.0
		df2 = inverse(transformation, df2)
		samples_to_show = minimum([nrow(df), nrow(df2), 8000])
		df_inds = rand(1:nrow(df), samples_to_show)
		df2_inds = rand(1:nrow(df2), samples_to_show)
		fig = compare_distributions(df[df_inds, variable_names], df2[df2_inds, variable_names]; 
								columns=latex_variable_names,
								figsize=(1000,800))
		Label(fig[1, 1, Top()], event_list[i], padding = (0, 0, 10, 0))
		temp_dir = mktempdir() do dir
			tempfile = joinpath(dir, "temp.pdf")
			save(tempfile, fig)
			append_pdf!(event_data_plot_location, tempfile, cleanup=true)
		end
	end

end


################ SELECTION INJECTIONS ########################

path = selection_function_samples_location

catalog = O1_O2_O3_sensitivity(path)
df_selections = get_injections(catalog)
analysis_time = get_analysis_time(catalog)
total_generated = get_total_generated(catalog)

cuts_kwargs = dict_eval(config_dict["selection"]["cuts"])
df_detected = implement_cuts(df_selections; cuts_kwargs...)

## Rename Columns
rename_maps = config_dict["selection"]["rename"] |> values |> collect .|> (x -> Meta.parse.(split(x[1], "=>"))) .|> (x->(x[1] => x[2]))
rename!(df_detected, rename_maps)


N_samples_to_fit_with = config_dict["selection"]["sampling"]["N_samples_to_fit_with"][1] |> Meta.parse |> eval
N_samples_per_kernel = config_dict["selection"]["sampling"]["N_samples_per_kernel"][1] |> Meta.parse |> eval
N_components = config_dict["selection"]["sampling"]["N_components"][1] |> Meta.parse |> eval

### Variable sets
variable_names = (config_dict["selection"]["variables"]["variable_names"] |> (x -> join(x, ",")) |> Meta.parse).args
variable_names = identity.(variable_names) ## Makes the type concrete

sampling_variable_names = (config_dict["selection"]["variables"]["sampling_variables"] |> (x -> join(x, ",")) |> Meta.parse).args
sampling_variable_names = identity.(sampling_variable_names) ## Makes the type concrete

analytical_variable_names = (config_dict["selection"]["variables"]["analytical_variables"] |> (x -> join(x, ",")) |> Meta.parse).args
analytical_variable_names = identity.(analytical_variable_names) ## Makes the type concrete

latex_variable_names = (config_dict["selection"]["variables"]["latex_names"] |> (x -> join(x, ",")) |> Meta.parse).args
latex_variable_names = identity.(eval.(latex_variable_names)) ## Makes the type concrete

### Add in prior
priors = config_dict["selection"]["prior"] |> values |> collect .|>  (x -> eval(Meta.parse(join(x, ","))))
prior = ProductPrior(priors)

evaluate!(df_detected, prior, :prior)

selection_samples = sample(df_detected, N_samples_to_fit_with, vcat(variable_names,[:prior]))

### Variable Truncation Limits
limits = join(config_dict["selection"]["variables"]["variable_truncation_limits"], ",") |> Meta.parse |> eval
limits = [Float64.(limit) for limit in limits]
a = [limit[1] for limit in limits]
b = [limit[2] for limit in limits]

if "prefittransformation" ∉ keys(config_dict["selection"])
	# Make an identity transformation	
	transformation = TruncatedGaussianMixtures.Transformation(
		variable_names,
		tuple,
		variable_names,
		tuple,
		[:prior]
	)
else
	transformed_variable_names = (config_dict["selection"]["prefittransformation"]["variable_names"] |> (x -> join(x, ",")) |> Meta.parse).args
	transformed_variable_names = identity.(transformed_variable_names)

	truncation_limits = join(config_dict["selection"]["prefittransformation"]["variable_truncation_limits"], ",") |> Meta.parse |> eval
	truncation_limits = [Float64.(limit) for limit in limits]
	truncation_a = [limit[1] for limit in truncation_limits]
	truncation_b = [limit[2] for limit in truncation_limits]

	forward_transformation = config_dict["selection"]["prefittransformation"]["transformation_function"] |> x->join(x,",") |> Meta.parse |> eval
	inverse_transformation = config_dict["selection"]["prefittransformation"]["inverse_transformation_function"] |> x->join(x,",") |> Meta.parse |> eval

	transformation = TruncatedGaussianMixtures.Transformation(
		variable_names,
		forward_transformation,
		transformed_variable_names,
		inverse_transformation,
		[:prior]
	)

	a = truncation_a
	b = truncation_b
end

var_map = Dict( :beta_max => :β_max, :delta_beta_rise => :dβ_rise, :delta_beta_fall => :dβ_relax, :N_beta_max => :N_high)

TGMM_annealing_kwargs = dict_eval(copy(conf._data["selection.truncatedgaussianmixturemodelparameters.annealingschedule"]); var_map=var_map)
β_schedule = AnnealingSchedule(;TGMM_annealing_kwargs...)


### Fitting events
var_map = Dict(:covariances => :cov, :tolerance => :tol)

TGMM_kwargs = dict_eval(copy(conf._data["selection.truncatedgaussianmixturemodelparameters"]); var_map=var_map)

mix, df = fit_gmm(selection_samples, N_components, a, b, transformation, β_schedule; TGMM_kwargs...)
the_dictionary = post_process_event_fit(df, mix, sampling_variable_names, analytical_variable_names, variable_names; N_samples_per_kernel=N_samples_per_kernel)

function save_dict_to_file(filename, the_dictionary; group_name="selection")
	h5open(filename, "w") do file
        group = create_group(file, group_name)
        save_dict_to_group(group, the_dictionary)
        write_attribute(group, "analysis_time", analysis_time)
        write_attribute(group, "total_generated", total_generated)
    end
end

save_dict_to_file(selection_data_output_location, the_dictionary)



################ DEBUG PLOTS #########################################
if "selection_data_plots" in keys(conf._data["datasources"])
	selection_data_plot_location = conf._data["datasources"]["selection_data_plots"][1][2:(end-1)]
	CairoMakie.activate!()

	# delete current pdf if it doesn't exist
	rm(selection_data_plot_location, force=true)

	samples_to_test = minimum([nrow(selection_samples), nrow(df), 8000])


	df2 = DataFrame(collect(rand(mix, samples_to_test)'), variable_names)
	fig = compare_distributions(selection_samples[rand(1:nrow(selection_samples), samples_to_test), variable_names], 
								df2[!, variable_names]; 
								columns=latex_variable_names,
								figsize=(1000,800))

	save(selection_data_plot_location, fig)
end
