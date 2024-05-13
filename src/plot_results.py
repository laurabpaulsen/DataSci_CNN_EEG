import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

# set font for all plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

def plot_results(results, output_dir):

    train_participants = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-08"]
    test_participants = ["sub-09", "sub-10", "sub-11", "sub-12", "sub-13"]

    models = train_participants + ["joint", "ensemble"] # add joint when done running
    participants = train_participants + test_participants

    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi = 300)

    for participant in participants:
        accuracies = []
        for model in models:
            key = f"{model}_{participant}"
            if key in results:
                accuracies.append(float(results[key]["accuracy"]))

            if participant in train_participants:
                marker = 'o'
            else:
                marker = 'x'

        ax.scatter(models, accuracies, marker=marker, label=participant)

    ax.legend(title="Test participant", ncol=2)
    
    ax.set_xlabel("Model")

    plt.savefig(output_dir)
        

if __name__ == "__main__":
    path = Path(__file__)

    # Initialize an empty dictionary to store the data
    results = {}

    # Open the text file and read its contents
    with open(path.parents[1] / "results" / "results.txt", 'r') as f:
        lines = f.readlines()

    # Iterate through each line in the file
    for line in lines:
        # Split the line by ':' to separate the key and value
        key, value = line.split(' : ')
        
        # Extract the model name from the key
        model_name = key.split('_')[0]
        
        # Remove curly braces and whitespace from the value string
        value = value.strip('{}\n')
        
        # Split the value string by ', ' to separate the key-value pairs
        pairs = value.split(', ')
        
        # Initialize an empty dictionary to store the key-value pairs
        results[key.strip()] = {}
        
        # Iterate through each key-value pair and add it to the dictionary
        for pair in pairs:
            # Split the pair by ':' to separate the key and value
            k, v = pair.split(': ')
            # Remove quotes from the keys and values
            k = k.strip(" '")
            v = v.strip(" '")
            # Add the key-value pair to the dictionary
            results[key.strip()][k] = v


    # Create an output directory for the plots
    output_dir = path.parents[1] / "fig"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    

    plot_results(results, output_dir / "mean_accuracies.png")