using Pkg
env_dir = join(split(@__FILE__, "/")[1:(end-2)], "/")
Pkg.activate(env_dir)

using Revise, DataFrames
using RingDB
using julia_scripts
using LaTeXStrings

database_folder = joinpath(homedir(), "Documents/Data/ringdb")
db = Database(database_folder)
event_list = readlines(joinpath(homedir(), "Documents/Data/selected_events_old.txt"))


transformation = Transformation(
	[:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift],
	(m1,q,χ1,χ2,z) -> (m1*to_chirp_mass(q)*(1+z), q,χ1,χ2,z),
	[:chirp_mass_det, :mass_ratio, :chi_1, :chi_2, :redshift],
	(M1,q,χ1,χ2,z) -> (M1/(to_chirp_mass(q)*(1+z)), q,χ1,χ2,z)
)

all_ranges = []
using ProgressMeter
@showprogress for eventname ∈ event_list
	event = Event(db, eventname)
	post = event.posteriors()
	select!(post, :, [:spin_1x, :spin_1y, :spin_1z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_1)
	select!(post, :, [:spin_2x, :spin_2y, :spin_2z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_2)
	post = forward(transformation, post)
	push!(all_ranges, Dict(c => ( minimum(post[!, c]), maximum(post[!, c]) ) for c ∈ names(post)))
end

# Initialize global minimum and maximum dictionaries
global_min = Dict{String, Float64}()
global_max = Dict{String, Float64}()

# Iterate through each dictionary in the list
for dict in all_ranges
    # Iterate through each key-value pair in the dictionary
    for (key, (min_val, max_val)) in dict
        # Update global minimum value for the key
        if haskey(global_min, key)
            global_min[key] = min(global_min[key], min_val)
        else
            global_min[key] = min_val
        end
        # Update global maximum value for the key
        if haskey(global_max, key)
            global_max[key] = max(global_max[key], max_val)
        else
            global_max[key] = max_val
        end
    end
end
