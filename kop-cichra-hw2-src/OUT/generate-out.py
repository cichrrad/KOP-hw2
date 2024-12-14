import os
import pandas as pd
import matplotlib.pyplot as plt

# Define a Nord color palette (my graphs deserve to be beatiful)
NORD_PALETTE = [
    "#88C0D0",  # Light Blue
    "#81A1C1",  # Blue
    "#5E81AC",  # Dark Blue
    "#BF616A",  # Red
    "#D08770",  # Orange
    "#EBCB8B",  # Yellow
    "#A3BE8C",  # Green
    "#B48EAD"   # Purple
]


def process_csv_files(directory):
    # Ensure output directories exist
    graphs_dir = os.path.join(directory, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    summary_rows = []
    
    # Define custom order for files
    custom_order = [
        "BATCH-20-91-M",
        "BATCH-20-91-N",
        "BATCH-20-91-Q",
        "BATCH-20-91-R",
        "BATCH-50-218-M",
        "BATCH-50-218-N",
        "BATCH-50-218-Q",
        "BATCH-50-218-R",
        "BATCH-75-325-M",
        "BATCH-75-325-N",
        "BATCH-75-325-Q",
        "BATCH-75-325-R"
    ]
    
    # Collect data for the merged graph
    avg_percentages = []  # List to store avg % values
    file_labels = []  # List to store file names (without .csv extension)

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            
            # Load the CSV file
            df = pd.read_csv(filepath)
            
            # Calculate the total number of entries and runs
            total_entries = len(df) - 1  # Exclude header row
            total_runs = total_entries * 100
            
            # Fetch values from the last row
            avg_percent = df.iloc[-1, 1]  # Second column
            exact_optimums = df.iloc[-1, 2]  # Third column
            blunders = df.iloc[-1, 3]  # Fourth column

            # Append data for the merged graph
            avg_percentages.append(float(avg_percent))
            file_labels.append(filename.replace(".csv", ""))  # Remove .csv extension for labels
            
            # Find best and worst performing entries by second and third columns
            best_avg_row = df.iloc[df.iloc[:-1, 1].idxmax()]  # Exclude last row
            worst_avg_row = df.iloc[df.iloc[:-1, 1].idxmin()]  # Exclude last row
            best_exact_row = df.iloc[df.iloc[:-1, 2].idxmax()]  # Exclude last row
            worst_exact_row = df.iloc[df.iloc[:-1, 2].idxmin()]  # Exclude last row

            # Save summary data for this file
            summary_rows.append({
                "File": filename,
                "Best Avg % Row": best_avg_row.to_dict(),
                "Worst Avg % Row": worst_avg_row.to_dict(),
                "Best Exact Optimums Row": best_exact_row.to_dict(),
                "Worst Exact Optimums Row": worst_exact_row.to_dict()
            })

            # Create the first bar graph (avg % of optimum vs 100%)
            plt.figure()
            plt.bar(["Avg %"], [float(avg_percent)], color=NORD_PALETTE[0], label=f"{float(avg_percent):.2f}%")
            plt.axhline(y=100, color=NORD_PALETTE[3], linestyle='--', label="100% Optimum")
            plt.title(f"Average % of Optimum - {filename}")
            plt.legend()
            plt.savefig(os.path.join(graphs_dir, f"{filename}_avg_percent.pdf"))
            plt.close()

            # Create the second graph as a pie chart
            plt.figure()
            slices = [exact_optimums, blunders, total_runs - (exact_optimums + blunders)]
            labels = ["Exact Optimums", "Blunders", "Other"]
            colors = [NORD_PALETTE[2], NORD_PALETTE[4], NORD_PALETTE[6]]
            plt.pie(slices, startangle=90, colors=colors)
            plt.title(f"Run Distribution - {filename}")
            plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), labels=[
                f"Exact Optimums: {exact_optimums}",
                f"Blunders: {blunders}",
                f"Total Runs: {total_runs}"
            ])
            plt.savefig(os.path.join(graphs_dir, f"{filename}_run_distribution.pdf"))
            plt.close()

    # Reorder data based on the custom order
    ordered_indices = [file_labels.index(label) for label in custom_order if label in file_labels]
    avg_percentages = [avg_percentages[i] for i in ordered_indices]
    file_labels = [file_labels[i] for i in ordered_indices]

    # Create the merged bar graph
    plt.figure(figsize=(10, 6))
    group_colors = {
        "20": NORD_PALETTE[0],
        "50": NORD_PALETTE[2],
        "75": NORD_PALETTE[4]
    }
    bar_colors = [group_colors[label.split('-')[1]] for label in file_labels]
    plt.bar(file_labels, avg_percentages, color=bar_colors)
    plt.axhline(y=100, color=NORD_PALETTE[3], linestyle='--', label="100% Optimum")
    plt.title("Average % of Optimum Comparison Across Datasets")
    plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
    plt.ylabel("Average % of Optimum")
    plt.legend()
    plt.tight_layout()  # Adjust layout for better fit
    plt.savefig(os.path.join(graphs_dir, "merged_avg_percent_comparison.pdf"))
    plt.close()

    # Save the summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(directory, "summary.csv"), index=False)

# Example usage
# process_csv_files("/path/to/directory")
