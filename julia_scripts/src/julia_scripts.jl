module julia_scripts

import TruncatedGaussianMixtures
import TruncatedGaussianMixtures: AbstractSchedule
import RingDB: AbstractTransformation, image_columns
using RingDB
using DataFrames, Distributions


function sample(df::DataFrame, N::Int)
	df[rand(1:nrow(df),N),:]
end

function sample(df::DataFrame, N::Int, columns)
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
	EM.mix, df_out
end



function TruncatedGaussianMixtures.fit_gmm(df::DataFrame, K, a, b, Tr::AbstractTransformation, S::AbstractSchedule; kwargs...)
	df2 = forward(Tr, df)
	EM = TruncatedGaussianMixtures.fit_gmm(collect(Matrix(df2[!, image_columns(Tr)])'), K, a, b, S; kwargs..., responsibilities=true)
	df_out = DataFrame(collect(hcat(EM.data...)'), names(df))

	## Make categorial assignment to different components of TGMM and output the dataframe with those assignments. 
	## We can then later use those to create groups
	assignments = [rand(Categorical(p)) for p ∈ EM.zⁿₖ]
	df_out[!, :components] = assignments
	EM.mix, df_out
end



using GLMakie, PairPlots, DataFrames
import Distributions
#fig = Figure(size=(600,600))
#X = collect(Matrix(df_detected)')

function make_new_plot!(fig, df, dfʼ, a, b; labels=nothing)
	c1 = Makie.wong_colors(0.5)[1]
	c2 = Makie.wong_colors(0.5)[2]
	empty!(fig)
    gs = GridLayout(fig[1,1])
    columns = names(df);
    axis = NamedTuple((Symbol(i), (; lims=(;low=aᵢ, high=bᵢ))) for (i,aᵢ,bᵢ) in zip(names(df), a, b))

    if labels == nothing
    	labels = Dict(Symbol(a)=>string(b) for (a,b) ∈ zip(columns, columns))
    end

    pairplot(gs, 
    	df => (PairPlots.Scatter(color=c1), PairPlots.Contourf(color=c1), PairPlots.MarginHist(color=c1)), 
    	dfʼ => (PairPlots.Scatter(color=c2), PairPlots.Contourf(color=c2), PairPlots.MarginHist(color=c2)),
    	axis=axis,
    	labels=labels
    )
    rowgap!(gs, 10)
	colgap!(gs, 10)
end


using Makie, GLMakie, PairPlots, DataFrames

make_df(x::Distributions.Distribution) = DataFrame(collect(rand(x, 8000)'), :auto)
make_df(x::DataFrames.DataFrame) = x

function compare_distributions(dist1, dist2; columns=:auto)
	X1 = make_df(dist1)
	X2 = DataFrame(collect(rand(dist2, 8000)'), names(X1))

	if columns isa Symbol
		columns = Dict([Symbol(s) => s for s in names(X1)])
	else
		columns = Dict([Symbol(s) => c for (s,c) in zip(names(X1), columns)])
	end

	@show columns


	fig = Figure()
	gs = GridLayout(fig[1,1])

	c1, c2 = Makie.wong_colors(0.5)[1:2]
	plot_attributes(color) = (
        PairPlots.Scatter(color=color),
        PairPlots.Contourf(color=color),

        # New:
        PairPlots.MarginHist(color=color),
        PairPlots.MarginConfidenceLimits(color=color),
    )

	pairplot(gs, PairPlots.Series(X1, label="Truth", color=c1, strokecolor=c1) => plot_attributes(c1),
			 	 PairPlots.Series(X2, label="Fit"  , color=c2, strokecolor=c2) => plot_attributes(c2),
			 	 labels = columns,
			 	 bodyaxis=(;
				            xlabelvisible=false,
				            xticksvisible=false,
				            xticklabelsvisible=false,
				            ylabelvisible=false,
				            yticksvisible=false,
				            yticklabelsvisible=false,
				        ),
				        diagaxis=(;
				            xlabelvisible=false,
				            xticksvisible=false,
				            xticklabelsvisible=false,
				            ylabelvisible=false,
				            yticksvisible=false,
				            yticklabelsvisible=false,
				        ))


	rowgap!(gs, 3)
	colgap!(gs, 3)

	fig
end



export sample
export fit_gmm
export compare_distributions, make_df, make_new_plot!

end # module julia_scripts
