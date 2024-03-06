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

event = Event(db, event_list[1])
post = event.posteriors()
select!(post, :, [:spin_1x, :spin_1y, :spin_1z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_1)
select!(post, :, [:spin_2x, :spin_2y, :spin_2z] => ByRow((x,y,z) -> sqrt(x^2 + y^2 + z^2)) => :chi_2)
post = sample(post, 8000, [:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift])

β = AnnealingSchedule(;β_max=1.5, dβ_rise=0.01, dβ_relax=0.01, N_high=20, N_post=200)

transformation = Transformation(
	[:mass_1_source, :mass_ratio, :chi_1, :chi_2, :redshift],
	(m1,q,χ1,χ2,z) -> (m1*to_chirp_mass(q)*(1+z),  q,χ1,χ2,z),
	[:chirp_mass_det, :mass_ratio, :chi_1, :chi_2, :redshift],
	(M1,q,χ1,χ2,z) -> (M1/(to_chirp_mass(q)*(1+z)),  q,χ1,χ2,z)
)

mix, df = fit_gmm(post, 5, [2.0, 0.0, 0.0, 0.0, 0.0], [100.0, 1.0, 1.0, 1.0, 3.0], transformation, β; progress=true, cov=:diag)
fig = compare_distributions(forward(transformation, post), mix, columns=[L"M_{chirp}^{det}", L"q", L"\chi_1", L"\chi_2", L"z"])
fig

