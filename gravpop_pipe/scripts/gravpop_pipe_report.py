import sys
import os
#from gravpop_pipe import *
from ..parser import *
from ..report import *
import pkg_resources
import os
import pprint

def get_file_path(file_name):
    # Use pkg_resources to get the file path
    return pkg_resources.resource_filename(__name__, file_name)

def run_based_on_file(file_path, samples_file_path, output_file):
    #from gravpop import *
    #from gravpop_pipe import *

    P = Parser(file_path)

    name = P.name

    P.load(samples_file_path)

    P4 = Parser("/Users/asadh/Documents/GitHub/gravpop_pipe/tests/model.ini")
    P4.load_lvk("/Users/asadh/Documents/Data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json")

    report = Report([P, P4], [name, "LVK"])

    report.save_image(filename=output_file);
    


#if __name__ == "__main__":
def main():
    # Check if a file path argument is provided
    if len(sys.argv) > 4:
        print("Usage: gravpop_run <ini_file_path> <samples_file_path> <output_file_path>")
    else:
        # Get the file path from the command line argument
        file_path = sys.argv[1]
        samples_file_path = sys.argv[2]
        output_file = "./lvk_comparison.pdf" 
        if len(sys.argv) == 4:
            output_file = sys.argv[3]
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"{file_path} not found.")
            return

        if not os.path.exists(samples_file_path):
            print(f"{samples_file_path} not found.")
            return

        run_based_on_file(file_path, samples_file_path, output_file=output_file)
