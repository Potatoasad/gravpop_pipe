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

def run_based_on_file(file_path):
    print(os.getcwd())

    P = Parser(file_path)

    name = P.name

    print(f"Running {name}")

    P.run()

    report = Report([P], [name])
    
    report.save_image();
    


#if __name__ == "__main__":
def main():
    # Check if a file path argument is provided
    if len(sys.argv) != 2:
        print("Usage: gravpop_run <file_path>")
    else:
        # Get the file path from the command line argument
        file_path = sys.argv[1]
        # Check if the file exists
        if not os.path.exists(file_path):
            print("File not found.")
            return
        run_based_on_file(file_path)
