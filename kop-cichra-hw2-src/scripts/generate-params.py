import json
import os
import numpy as np
from itertools import product


def generate_configs(config_path, out_dir, num_values=5):
    # Load the input configuration
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Fixed parameters (where boundaries are the same)
    fixed_params = {}
    param_values = {}

    # Separate fixed parameters and interpolated parameters
    for param, range_values in config.items():
        if range_values[0] == range_values[1]:
            # Fix parameter to the single value
            fixed_params[param] = range_values[0]
        else:
            # Generate interpolated values
            if param == "max_stagnation":
                # Integers: Generate evenly spaced integers
                param_values[param] = np.linspace(
                    range_values[0], range_values[1], num=num_values, dtype=int
                ).tolist()
            else:
                # Floats: Generate evenly spaced floats
                param_values[param] = np.linspace(
                    range_values[0], range_values[1], num=num_values
                ).tolist()

    # Create combinations of parameter values
    combinations = list(product(*param_values.values()))
    param_keys = list(param_values.keys())

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save each configuration as a JSON file
    for idx, combo in enumerate(combinations):
        config_dict = dict(zip(param_keys, combo))
        config_dict.update(fixed_params)  # Add fixed parameters
        output_path = os.path.join(out_dir, f"config_{idx + 1}.json")
        with open(output_path, 'w') as outfile:
            json.dump(config_dict, outfile, indent=4)

    print(f"Generated {len(combinations)} configurations in {out_dir}")


# Example usage
if __name__ == "__main__":
    config_file = "path/to/file"  # Input JSON file
    output_directory = "path/to/dir"  # Output directory for configurations
    generate_configs(config_file, output_directory)
