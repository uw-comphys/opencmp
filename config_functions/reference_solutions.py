from .base_config_functions import ConfigFunctions
import ngsolve as ngs
from typing import Dict, Union


class RefSolFunctions(ConfigFunctions):
    """
    Class to hold the initial condition functions.
    """

    def __init__(self, config_rel_path: str, t_param: ngs.Parameter = ngs.Parameter(0.0),
                 new_variables: Dict[str, Union[float, ngs.CoefficientFunction, ngs.GridFunction]] = {}) -> None:
        super().__init__(config_rel_path, t_param)

        # Load the reference solution dict from the reference solution configfile.
        try:
            ref_sols, ref_sols_re_parse = self.config.get_one_level_dict('REFERENCE SOLUTIONS', self.t_param, new_variables)
        except KeyError:
            # No options specified for L2_error.
            ref_sols = {}
            ref_sols_re_parse = {}

        try:
            metrics, metrics_re_parse = self.config.get_one_level_dict('METRICS', self.t_param, new_variables, all_str=True)
        except KeyError:
            # No options specified for other metrics.
            metrics = {}
            metrics_re_parse = {}

        self.ref_sol_dict = {'ref_sols': ref_sols, 'metrics': metrics}
        self.ref_sol_re_parse_dict = {'ref_sols': ref_sols_re_parse, 'metrics': metrics_re_parse}

    def set_ref_solution(self, fes: ngs.FESpace, model_components: Dict[str, int]):
        """
        Function to load the reference solutions from their configfile into a dict, including loading any saved
        gridfunctions.

        Args:
            fes: The finite element space of the model.
            model_components: Maps between variable names and their component in the finite element space.

        Returns:
            ref_sol_dict: Dict of reference solutions.
        """

        # Turn 'all' into a separate reference solution for each variable. Assuming that anytime 'all' is used the
        # reference solution is being loaded in from file since there is no way to parse closed form expressions for
        # multiple gridfunction components.
        if 'all' in self.ref_sol_dict['ref_sols'].keys():
            val = self.ref_sol_dict['ref_sols'].pop('all')
            for var in model_components.keys():
                self.ref_sol_dict['ref_sols'][var] = val

        for var, val in self.ref_sol_dict['ref_sols'].items():
            if isinstance(val, str):
                # Need to load a gridfunction from file.

                # Check that the file exists.
                val = self._find_rel_path_for_file(val)

                component = model_components[var]
                if component is None:
                    # Solution for entire finite element space.
                    gfu_val = ngs.GridFunction(fes)
                    gfu_val.Load(val)

                else:
                    # Use a component of the finite element space.
                    gfu_val = ngs.GridFunction(fes.components[component])
                    gfu_val.Load(val)

                # Replace the value in the BC dictionary.
                self.ref_sol_dict['ref_sols'][var] = gfu_val

            else:
                # Already parsed
                pass

        return self.ref_sol_dict

    def update_ref_solutions(self, t_param: ngs.Parameter,
                                   updated_variables: Dict[
                                       str, Union[float, ngs.CoefficientFunction, ngs.GridFunction]]) \
            -> None:
        """
        Function to update the reference solutions with new values of the model_variables.

        Args:
            t_param: The time parameter.
            updated_variables: Dictionary containing the model variables and their updated values.
        """

        for k1, v1 in self.ref_sol_re_parse_dict.items():
            self.ref_sol_dict[k1] = self.re_parse(self.ref_sol_dict[k1], self.ref_sol_re_parse_dict[k1], t_param, updated_variables)

