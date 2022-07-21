from opencmp.post_processing import sol_to_vtu_direct
from opencmp.config_functions import ConfigParser

path_to_config_file = "config"
path_to_output_director = "output/"

sol_to_vtu_direct(ConfigParser(path_to_config_file), path_to_output_director)
