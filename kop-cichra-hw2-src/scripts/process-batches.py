import os
import csv
import pandas as pd

def process_batch_directories(in_dir):
    # Ensure the root directory exists
    if not os.path.exists(in_dir):
        print(f"Error: The directory {in_dir} does not exist.")
        return

    # Get all BATCH directories
    batch_dirs = [d for d in os.listdir(in_dir) if d.startswith("BATCH-")]

    damaged_runs = []  # To log directories with damaged outputs

    for batch_dir in batch_dirs:
        batch_path = os.path.join(in_dir, batch_dir)
        if not os.path.isdir(batch_path):
            continue

        # Create a summary CSV for the current BATCH directory
        summary_rows = []

        fitness_test_dirs = [d for d in os.listdir(batch_path) if d.startswith("fitness_test_wuf")]

        for fitness_dir in fitness_test_dirs:
            fitness_path = os.path.join(batch_path, fitness_dir)

            if not os.path.isdir(fitness_path):
                continue

            # Check the outputs directory for damaged runs
            outputs_path = os.path.join(fitness_path, "outputs")
            damaged = False

            if os.path.exists(outputs_path) and os.path.isdir(outputs_path):
                output_files = [f for f in os.listdir(outputs_path) if f.endswith(".csv")]

                if len(output_files) == 2:
                    for output_file in output_files:
                        output_file_path = os.path.join(outputs_path, output_file)
                        try:
                            with open(output_file_path, mode='r') as file:
                                reader = csv.reader(file)
                                rows = list(reader)
                                if len(rows) <= 1:  # Only header row or empty
                                    damaged = True
                                    break
                        except Exception as e:
                            damaged = True
                            print(f"Warning: Failed to process {output_file_path}. Error: {e}")
                            break
                else:
                    damaged = True
            else:
                damaged = True

            if damaged:
                damaged_runs.append({"Batch": batch_dir, "Fitness Directory": fitness_dir})
                continue

            # Collect all .csv files in the current fitness_test directory
            csv_files = [f for f in os.listdir(fitness_path) if f.endswith(".csv")]

            percentages = []
            exact_optimums = 0
            blunders = 0
            formula_name = None

            for csv_file in csv_files:
                csv_path = os.path.join(fitness_path, csv_file)

                try:
                    with open(csv_path, mode='r') as file:
                        reader = csv.DictReader(file)

                        for row in reader:
                            try:
                                if formula_name is None:
                                    formula_name = row.get("Formula")

                                percentage = float(row.get("Percentage of Optimal", 0))
                                if percentage < 0 or percentage > 100:
                                    raise ValueError(f"Invalid percentage: {percentage}")

                                percentages.append(percentage)

                                if percentage == 100.0:
                                    exact_optimums += 1
                                elif percentage == 0.0:
                                    blunders += 1

                            except ValueError as ve:
                                print(f"Warning: Skipping invalid row in {csv_path}. Error: {ve}")
                                continue

                except Exception as e:
                    print(f"Warning: Failed to process {csv_path}. Error: {e}")
                    continue

            if percentages:
                avg_percentage = sum(percentages) / len(percentages)
                print(f"Processed {fitness_dir}: Average={avg_percentage}, Exact Optimums={exact_optimums}, Blunders={blunders}")
                summary_rows.append({
                    "Formula": formula_name,
                    "avg % of optimum found": avg_percentage,
                    "exact optimums found": exact_optimums,
                    "blunders": blunders
                })
            else:
                print(f"Warning: No valid data found in {fitness_dir}.")

        # Add summary statistics to the bottom of the CSV
        if summary_rows:
            total_avg_percentage = sum(row["avg % of optimum found"] for row in summary_rows) / len(summary_rows)
            total_exact_optimums = sum(row["exact optimums found"] for row in summary_rows)
            total_blunders = sum(row["blunders"] for row in summary_rows)

            summary_rows.append({
                "Formula": "TOTAL",
                "avg % of optimum found": total_avg_percentage,
                "exact optimums found": total_exact_optimums,
                "blunders": total_blunders
            })

        # Write summary to a CSV file
        output_csv_path = os.path.join(in_dir, f"{batch_dir}.csv")
        try:
            df = pd.DataFrame(summary_rows)
            df.to_csv(output_csv_path, index=False)
            print(f"Summary saved: {output_csv_path}")
        except Exception as e:
            print(f"Error: Failed to save summary for {batch_dir}. Error: {e}")

    # Write damaged runs to a separate CSV file
    damaged_csv_path = os.path.join(in_dir, "damaged_runs.csv")
    try:
        df_damaged = pd.DataFrame(damaged_runs)
        df_damaged.to_csv(damaged_csv_path, index=False)
        print(f"Damaged runs saved: {damaged_csv_path}")
    except Exception as e:
        print(f"Error: Failed to save damaged runs log. Error: {e}")

# Example usage:
#process_batch_directories("path/to/dir")
