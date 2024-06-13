module julia_scripts

import TruncatedGaussianMixtures
import TruncatedGaussianMixtures: AbstractSchedule
using TruncatedGaussianMixtures
#import RingDB: AbstractTransformation, image_columns
using RingDB
using DataFrames, Distributions
using HDF5

""""
function sample(df::AbstractDataFrame, N::Int)
	df[rand(1:nrow(df),N),:]
end

function sample(df::AbstractDataFrame, N::Int, columns)
	df[rand(1:nrow(df),N), columns]
end


function TruncatedGaussianMixtures.fit_gmm(df::DataFrame, K, a, b; kwargs...)
	TruncatedGaussianMixtures.fit_gmm(collect(Matrix(df)'), K, a, b; kwargs...)
end

function TruncatedGaussianMixtures.fit_gmm(df::DataFrame, K, a, b, S::AbstractSchedule; kwargs...)
	TruncatedGaussianMixtures.fit_gmm(collect(Matrix(df)'), K, a, b, S; kwargs...)
end

function TruncatedGaussianMixtures.fit_gmm(df::DataFrame, K, a, b, Tr::AbstractTransformation; kwargs...)
	df2 = forward(Tr, df)
	EM = TruncatedGaussianMixtures.fit_gmm(collect(Matrix(df2[!, image_columns(Tr)])'), K, a, b; kwargs..., responsibilities=true)
	df_out = DataFrame(collect(hcat(EM.data...)'), names(df))

	## Make categorial assignment to different components of TGMM and output the dataframe with those assignments. 
	## We can then later use those to create groups
	assignments = [rand(Categorical(p)) for p ∈ EM.zⁿₖ]
	df_out[!, :components] = assignments
	for col ∈ Tr.ignore_columns
		df_out[!, col] = df[!, col]
	end
	#if (:components ∉ Tr.ignore_columns)
	#	push!(Tr.ignore_columns, :components)
	#end
	df_out = inverse(Tr, df_out)
	df_out[!, :components] = assignments
	EM.mix, df_out
end



function TruncatedGaussianMixtures.fit_gmm(df::DataFrame, K, a, b, Tr::AbstractTransformation, S::AbstractSchedule; kwargs...)
	df2 = forward(Tr, df)
	EM = TruncatedGaussianMixtures.fit_gmm(collect(Matrix(df2[!, image_columns(Tr)])'), K, a, b, S; kwargs..., responsibilities=true)
	df_out = DataFrame(collect(hcat(EM.data...)'), image_columns(Tr))

	## Make categorial assignment to different components of TGMM and output the dataframe with those assignments. 
	## We can then later use those to create groups
	assignments = [rand(Categorical(p)) for p ∈ EM.zⁿₖ]
	df_out[!, :components] = assignments
	for col ∈ Tr.ignore_columns
		df_out[!, col] = df[!, col]
	end
	#if (:components ∉ Tr.ignore_columns)
	#	push!(Tr.ignore_columns, :components)
	#end
	df_out = inverse(Tr, df_out)
	df_out[!, :components] = assignments
	EM.mix, df_out
end
"""

include("./post_process_fit.jl")
include("./plotting_helper.jl")

export sample
export fit_gmm
export compare_distributions, make_df, make_new_plot!
export post_process_event_fit, grab_event_data, save_dict_to_group, save_list_of_dicts_to_hdf5

end # module julia_scripts
