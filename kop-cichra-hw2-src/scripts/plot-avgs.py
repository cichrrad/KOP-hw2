import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_and_average_data(data_dir, filename):
    """
    Load and average data from all log files matching the given filename.
    :param data_dir: Directory containing the log files.
    :param filename: Base filename (e.g., "problem_name") to find corresponding logs.
    :return: A pandas DataFrame with averaged data for weight and clauses satisfied.
    """
    # Collect all matching log files
    log_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.startswith(f"{filename}_") and f.endswith(".csv") and not f.startswith("summary")
    ]

    if not log_files:
        raise ValueError(f"No log files found for {filename} in {data_dir}.")

    print(f"Found {len(log_files)} log files for {filename}.")

    # Load and aggregate data
    dfs = []
    for file in log_files:
        try:
            df = pd.read_csv(file)
            # Drop the 'Configuration' column since it's not numeric
            df = df.drop(columns=["Configuration"])
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        raise ValueError(f"Failed to load any valid log files for {filename}.")

    # Combine all data and average
    combined_df = pd.concat(dfs, axis=0).groupby("Iteration").mean().reset_index()
    return combined_df

def load_summary_data(data_dir, filename):
    """
    Load and calculate average best weight and determine the optimum weight from summary files.
    :param data_dir: Directory containing the summary files.
    :param filename: Base filename (e.g., "problem_name") to find corresponding summaries.
    :return: A tuple (average_best_weight, optimum_weight).
    """
    # Collect all matching summary files
    summary_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.startswith(f"summary_{filename}_") and f.endswith(".csv")
    ]

    if not summary_files:
        print(f"No summary files found for {filename} in {data_dir}.")
        return None, None

    print(f"Found {len(summary_files)} summary files for {filename}.")

    # Load and aggregate summary data
    best_weights = []
    optimum_weight = None
    for file in summary_files:
        try:
            df = pd.read_csv(file)
            best_weights.extend(df["Best Weight Found"].dropna().tolist())
            if "Optimal Weight" in df and not pd.isna(df["Optimal Weight"].iloc[0]):
                optimum_weight = df["Optimal Weight"].iloc[0]
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not best_weights:
        print(f"No valid 'Best Weight Found' data for {filename}.")
        return None, optimum_weight

    average_best_weight = sum(best_weights) / len(best_weights)
    return average_best_weight, optimum_weight

def plot_and_save_averages(data_dir, filename, out_dir):
    """
    Plot averaged weight and clauses satisfied over iterations for a given problem.
    Save the averaged data to CSV files.
    :param data_dir: Directory containing the log files.
    :param filename: Base filename (e.g., "problem_name") to find corresponding logs.
    :param out_dir: Directory to save the output plots and CSV files.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Load and average data
    avg_data = load_and_average_data(data_dir, filename)

    # Save averaged data to CSV files
    weights_csv_path = os.path.join(out_dir, f"{filename}_averaged_weights.csv")
    clauses_csv_path = os.path.join(out_dir, f"{filename}_averaged_clauses.csv")
    avg_data[["Iteration", "Current Weight"]].to_csv(weights_csv_path, index=False)
    avg_data[["Iteration", "Satisfied Clauses"]].to_csv(clauses_csv_path, index=False)
    print(f"Saved averaged weights to {weights_csv_path}")
    print(f"Saved averaged clauses to {clauses_csv_path}")

    # Load summary data for average best weight and optimum weight
    avg_best_weight, optimum_weight = load_summary_data(data_dir, filename)

    # Plot average weight
    plt.figure(figsize=(10, 6))
    plt.plot(avg_data["Iteration"], avg_data["Current Weight"], label="Average Weight")
    #if avg_best_weight is not None:
    #    plt.axhline(y=avg_best_weight, color="green", linestyle="--", label="Average Best Weight")
    if optimum_weight is not None:
        plt.axhline(y=optimum_weight, color="red", linestyle="--", label="Optimum Weight")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.title(f"Average Weight vs. Iteration for {filename}")
    plt.legend()
    weight_plot_path = os.path.join(out_dir, f"{filename}_avg_weight.pdf")
    plt.savefig(weight_plot_path)
    plt.close()
    print(f"Saved weight plot to {weight_plot_path}")

    # Plot average clauses satisfied
    plt.figure(figsize=(10, 6))
    plt.plot(avg_data["Iteration"], avg_data["Satisfied Clauses"], label="Average Clauses Satisfied", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Clauses Satisfied")
    plt.title(f"Average Clauses Satisfied vs. Iteration for {filename}")
    plt.legend()
    clauses_plot_path = os.path.join(out_dir, f"{filename}_avg_clauses.pdf")
    plt.savefig(clauses_plot_path)
    plt.close()
    print(f"Saved clauses satisfied plot to {clauses_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize fitness data from annealing logs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the log files.")
    parser.add_argument("--filename", type=str, required=True, help="Base filename of the problem (e.g., 'problem_name').")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the output plots and CSV files.")

    args = parser.parse_args()

    plot_and_save_averages(data_dir=args.data_dir, filename=args.filename, out_dir=args.out_dir)
