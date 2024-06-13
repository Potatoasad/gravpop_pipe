

function post_process_event_fit(df, mix, sampling_variables, analytical_variables, variables; N_samples_per_kernel=1000)
	### Sampling Variables
	sampling_data = []
	the_dict = Dict()
	for variable ∈ sampling_variables
		the_dict[variable] = Vector{Float64}[]
	end
	for variable ∈ analytical_variables
		the_dict[variable] = Vector{Float64}[]
	end
	the_dict[:prior] = Vector{Float64}[]

	for df_component ∈ groupby(df, :components)
		X = sample(df_component, N_samples_per_kernel)
		for variable ∈ sampling_variables
			push!(the_dict[variable], X[!, variable])
		end
		for variable ∈ analytical_variables
			push!(the_dict[variable], X[!, variable])
		end
		push!(the_dict[:prior], X[!, :prior])
	end

	for variable ∈ sampling_variables
		the_dict[variable] = hcat(the_dict[variable]...)
	end

	for variable ∈ analytical_variables
		the_dict[variable] = hcat(the_dict[variable]...)
	end
	the_dict[:prior] = hcat(the_dict[:prior]...)
	the_dict[:weights] = mix.prior.p


	### Analytical Variables
	index_of_analytical_variables = Dict(var => findall(x->(x==var), variables)[1] for var ∈ analytical_variables)
	for variable ∈ analytical_variables
		new_mu_name = Symbol("$(variable)_mu_kernel")
		new_sigma_name = Symbol("$(variable)_sigma_kernel")
		i = index_of_analytical_variables[variable]
		the_dict[new_mu_name] = [ comp.normal.μ[i] for comp ∈ mix.components]
		the_dict[new_sigma_name] = [ sqrt(comp.normal.Σ[i,i]) for comp ∈ mix.components]
	end

	### GMM storage
	gmm_dict = Dict()
	gmm_dict[:means] = hcat([comp.normal.μ for comp ∈ mix.components]...)
	gmm_dict[:covariances] = stack([comp.normal.Σ for comp ∈ mix.components])
	gmm_dict[:weights] = mix.prior.p
	the_dict[:gmm] = gmm_dict

	the_dict
end



function grab_event_data(event, prior, transformation, β_schedule, settings)
	# Unpack the dictionary into variables
	N_samples_to_fit_with = settings[:N_samples_to_fit_with]
	N_samples_per_kernel = settings[:N_samples_per_kernel]
	N_components = settings[:N_components]
	variables = settings[:variables]
	sampling_variables = settings[:sampling_variables]
	analytical_variables = settings[:analytical_variables]
	a = settings[:a]
	b = settings[:b]


	### Get Event and add variables
	post = event.posteriors()
	select!(post, :, [:spin_1x, :spin_1y, :spin_1z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_1)
	select!(post, :, [:spin_2x, :spin_2y, :spin_2z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_2)

	post = sample(post, N_samples_to_fit_with, variables)

	### Evaluate Prior on event posterior samples
	evaluate!(post, prior, :prior)

	### Set up gaussian mixture model
	mix, df = fit_gmm(post, N_components, a, b, transformation, β_schedule; progress=false, cov=:diag)

	print(names(df))

	### Post process TGMM for hybrid scheme
	event_data_dictionary = post_process_event_fit(df, mix, sampling_variables, analytical_variables, variables; N_samples_per_kernel=N_samples_per_kernel)

	event_data_dictionary[:julia_data] = Dict(:gmm => mix, :df => df)	
	event_data_dictionary
end

function save_list_of_dicts_to_hdf5(data_list::Vector{T}, event_list::Vector{String}, filename::String) where {T <: Any}
    if length(data_list) != length(event_list)
        error("Length of data_list and event_list must be the same")
    end
    
    h5open(filename, "w") do file
        for i in 1:length(data_list)
            group = create_group(file, event_list[i])
            save_dict_to_group(group, data_list[i])
        end
    end
end

function save_dict_to_group(group, data::Dict{K, T}) where {K<:Any, T<:Any}
    for (key, value) in data
        if isa(value, AbstractArray)
            dset = create_dataset(group, string(key), datatype(value), size(value))
            write(dset, value)
        elseif isa(value, Dict)
            sub_group = create_group(group, string(key))
            save_dict_to_group(sub_group, value)
        else
            print("Unsupported data type for key $key")
        end
    end
end

