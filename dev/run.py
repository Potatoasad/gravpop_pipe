from gravpop_pipe import *


if __name__ == '__main__':
	P = Parser("/Users/asadh/Documents/GitHub/gravpop_pipe/tests/hybrid_model.ini")

	P.run()

	P4 = Parser("/Users/asadh/Documents/GitHub/gravpop_pipe/tests/model.ini")
	P4.load_lvk("/Users/asadh/Documents/Data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json")

	report = Report([P, P4], ["gravpop", "LVK"])

	report.save_image();