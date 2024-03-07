using Makie, GLMakie, PairPlots, DataFrames
import Distributions

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


make_df(x::Distributions.Distribution; columns=:auto, N=8000) = DataFrame(collect(rand(x, N)'), columns)
make_df(x::DataFrames.DataFrame; columns=:auto, N=8000) = rename(x[rand(1:nrow(x), N), :], Dict([col => label for (col, label) ∈ zip(names(x), columns)]))

function compare_distributions(dist1, dist2; columns=:auto)
	X1 = make_df(dist1; columns=columns, N=8000)
	X2 = make_df(dist2; columns=columns, N=8000)#DataFrame(collect(rand(dist2, 8000)'), names(X1))

	@show names(X1), names(X2)

	if columns isa Symbol
		columns = Dict([Symbol(s) => s for s in names(X1)])
	else
		columns = Dict([Symbol(s) => c for (s,c) in zip(names(X1), columns)])
	end

	rename!(X1, Dict([x => Symbol(x) for x ∈ names(X1)]))
	rename!(X2, Dict([x => Symbol(x) for x ∈ names(X2)]))


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
			 	 labels = columns)


	rowgap!(gs, 3)
	colgap!(gs, 3)

	fig
end
