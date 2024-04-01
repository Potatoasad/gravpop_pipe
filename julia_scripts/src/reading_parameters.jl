using Pkg
env_dir = join(split(@__FILE__, "/")[1:(end-2)], "/")
Pkg.activate(env_dir)

using ConfParser
using ArgParse
using RingDB


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

event_name_list			   = conf._data["datasources"]["event_list"][1][2:(end-1)]
selection_function_samples = conf._data["datasources"]["selection_function_samples"][1][2:(end-1)]
database_location 		   = conf._data["datasources"]["database_location"][1][2:(end-1)]

db = Database(database_location)
events = readlines(event_name_list)

