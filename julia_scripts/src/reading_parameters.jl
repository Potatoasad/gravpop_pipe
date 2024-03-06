using ConfParser
using RingDB

conf = ConfParse("test/test.ini")
parse_conf!(conf)

event_name_list			   = conf._data["datasources"]["event_list"][1][2:(end-1)]
selection_function_samples = conf._data["datasources"]["selection_function_samples"][1][2:(end-1)]
database_location 		   = conf._data["datasources"]["database_location"][1][2:(end-1)]

db = Database(database_location)
events = readlines(event_name_list)