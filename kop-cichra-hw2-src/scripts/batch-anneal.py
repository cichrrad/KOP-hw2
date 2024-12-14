import os
import argparse
import csv
import random
import json
from mwsat import WeightedSATParser, SATSolverScored2

def parse_optima(optima_file):
    """
    Parse the optima file to extract optimal weights and configurations for given formulas.
    :param optima_file: Path to the file containing optimal solutions.
    :return: A dictionary mapping filenames to (optimal_weight, optimal_configuration).
    """
    optima = {}
    with open(optima_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # Skip malformed lines

            filename = parts[0]  # The first column is the filename
            try:
                optimal_weight = int(parts[1])  # The second column is the optimal weight
                optimal_configuration = [
                    True if int(val) > 0 else False for val in parts[2:-1]
                ]  # The rest is the configuration (convert to booleans), exclude trailing `0`
                optima[filename] = (optimal_weight, optimal_configuration)
            except ValueError:
                print(f"Error parsing line: {line}")  # Log and skip malformed lines

    return optima

def run_annealing(data_dir, optima_file, out_dir, batch_run, **annealing_params):
    """
    Run annealing for all formulas in the specified directory and compare results with optima.
    :param data_dir: Directory containing SAT formula files.
    :param optima_file: File with optimal solutions for formulas.
    :param out_dir: Directory to save annealing logs.
    :param batch_run: The current batch run number (used for log filenames).
    :param annealing_params: Parameters for the annealing process.
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    optima = parse_optima(optima_file)
    results = []

    for formula_file in os.listdir(data_dir):
        if not formula_file.endswith('.mwcnf'):
            continue  # Skip non-CNF files
        # Strip extension for matching with optima
        formula_name = os.path.splitext(formula_file)[0]

        print(f"Processing formula: {formula_file} (Batch Run: {batch_run})")
        parser = WeightedSATParser(os.path.join(data_dir, formula_file))
        parser.parse_file()

        solver = SATSolverScored2(problem=parser)
        log_file = os.path.join(out_dir, f"{formula_name}_{batch_run}.csv")
        solver.csv_log_path = log_file  # Custom log file name

        # Run annealing
        best_solution, best_weight = solver.solve(**annealing_params)

        # Compare with optima (if available)
        optimal_weight, optimal_configuration = optima.get(formula_name[1:], (None, None))
        percentage_of_optimal = (
            (best_weight / optimal_weight) * 100 if optimal_weight else None
        )

        results.append({
            "Formula": formula_file,
            "Best Weight Found": best_weight,
            "Optimal Weight": optimal_weight,
            "Percentage of Optimal": percentage_of_optimal,
            "Log File": log_file,
        })

        print(f"Best Weight Found: {best_weight}")
        if optimal_weight:
            print(f"Optimal Weight: {optimal_weight} ({percentage_of_optimal:.2f}%)")

        # Save a summary file for this formula and batch run
        summary_file = os.path.join(out_dir, f"summary_{formula_name}_{batch_run}.csv")
        with open(summary_file, "w", newline="") as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=["Formula", "Best Weight Found", "Optimal Weight", "Percentage of Optimal", "Log File"])
            csvwriter.writeheader()
            csvwriter.writerows(results)

        print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulated annealing on SAT formulas and compare with optima.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing SAT formula files.")
    parser.add_argument("--optima_file", type=str, required=True, help="File containing optimal solutions for formulas.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save annealing logs.")
    parser.add_argument("--batch_runs", type=int, default=100, help="Number of batch runs to perform.")
    parser.add_argument("--T0", type=float, default=17.5, help="Initial temperature for annealing.")
    parser.add_argument("--alpha", type=float, default=0.99, help="Cooling rate for annealing.")
    parser.add_argument("--max_iterations", type=int, default=100000, help="Maximum number of annealing iterations.")
    parser.add_argument("--T_min", type=float, default=0.001, help="Minimum temperature for annealing.")
    parser.add_argument("--max_stagnation", type=int, default=30, help="Maximum stagnation iterations before reheating.")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file for annealing parameters.")

    args = parser.parse_args()

    # Load parameters from JSON config if provided
    config_params = {}
    if args.config:
        with open(args.config, 'r') as config_file:
            config_params = json.load(config_file)

    # Combine parameters from JSON config and command line (command line takes precedence)
    annealing_params = {
        "T0": config_params.get("T0", args.T0),
        "alpha": config_params.get("alpha", args.alpha),
        "max_iterations": config_params.get("max_iterations", args.max_iterations),
        "T_min": config_params.get("T_min", args.T_min),
        "max_stagnation": config_params.get("max_stagnation", args.max_stagnation),
    }

    # Perform the specified number of batch runs
    for batch_run in range(1, args.batch_runs + 1):
        run_annealing(
            data_dir=args.data_dir,
            optima_file=args.optima_file,
            out_dir=args.out_dir,
            batch_run=batch_run,
            **annealing_params,
        )
