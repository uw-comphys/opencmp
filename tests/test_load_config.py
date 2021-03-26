import pytest
from config_functions.load_config import parse_str, convert_str_to_dict, load_coefficientfunction_into_gridfunction
import pyparsing
import ngsolve as ngs

@pytest.fixture()
def presets_convert_str_to_dict():
    config_filename = 'tests/example_config'
    correct_dict = {'a': 12, 'b': 'mesh.vol', 'c': 1}
    correct_re_parse_dict = {'c': 'u+p'}
    new_variables = {'u': 0, 'p': 1}
    filetypes = ['.vol']

    return config_filename, correct_dict, correct_re_parse_dict, new_variables, filetypes

class TestParseStr:
    """ Class to test parse_to_str. """
    def test_1(self):
        """ Check that strings of the specified filetypes get kept as strings. """
        filetype = '.vol'
        input_str = 'mesh' + filetype
        output_str, variable_eval = parse_str(input_str, ngs.Parameter(0.0), {}, [filetype])

        assert output_str == input_str # The input string should not have been parsed.
        assert not variable_eval       # The input string should not be flagged for re-parsing.

    def test_2(self):
        """
        Check that strings of a non-specified filetype are parsed. This should raise an error since they wouldn't be
        parseable.
        """
        filetype = '.vol'
        unspecified_filetype = '.stl'
        input_str = 'mesh' + unspecified_filetype

        with pytest.raises(pyparsing.ParseException): # Expect a parser error.
            parse_str(input_str, ngs.Parameter(0.0), {}, [filetype])

    def test_3(self):
        """ Check that generic strings are parsed. """
        input_str = '3^2 + 6'            # Need to use a basic expression, can't directly compare coefficientfunctions.
        parsed_str = 3.0**2 + 6.0
        output_str, variable_eval = parse_str(input_str, ngs.Parameter(0.0))

        assert output_str == parsed_str  # The input string was parsed correctly.

    def test_4(self):
        """ Check that non-string inputs do not get parsed and just get passed through. """
        input_obj = [1, 2, 3]
        output_obj, variable_eval = parse_str(input_obj, ngs.Parameter(0.0))

        assert output_obj == input_obj  # The input object should not have been parsed.
        assert not variable_eval        # The input object should not be flagged for re-parsing.

class TestConvertStrToDict:
    """ Class to test convert_str_to_dict. """
    def test_1(self, load_configfile, presets_convert_str_to_dict):
        """ Check normal functioning. """
        config_section = 'TEST LOAD CONFIG'
        config_key = 'test_1'
        config_filename, correct_dict, correct_re_parse_dict, new_variables, filetypes = presets_convert_str_to_dict

        config = load_configfile(config_filename)
        input_str = config[config_section][config_key]
        output_dict, output_re_parse_dict = convert_str_to_dict(input_str, ngs.Parameter(0.0), new_variables, filetypes)

        assert output_dict == correct_dict                   # Check that the parameter dictionary is correct.
        assert output_re_parse_dict == correct_re_parse_dict # Check that the re-parse dictionary is correct.

    def test_2(self, load_configfile, presets_convert_str_to_dict):
        """ Check that newlines are necessary separators. """
        config_section = 'TEST LOAD CONFIG'
        config_key = 'test_2'
        config_filename, correct_dict, correct_re_parse_dict, new_variables, filetypes = presets_convert_str_to_dict

        config = load_configfile(config_filename)
        input_str = config[config_section][config_key]

        with pytest.raises(Exception):  # Expect an error.
            convert_str_to_dict(input_str, ngs.Parameter(0.0), new_variables, filetypes)

    def test_3(self, load_configfile, presets_convert_str_to_dict):
        """ Check that -> is a necessary separator. """
        config_section = 'TEST LOAD CONFIG'
        config_key = 'test_3'
        config_filename, correct_dict, correct_re_parse_dict, new_variables, filetypes = presets_convert_str_to_dict

        config = load_configfile(config_filename)
        input_str = config[config_section][config_key]

        with pytest.raises(Exception):  # Expect an error.
            convert_str_to_dict(input_str, ngs.Parameter(0.0), new_variables, filetypes)
