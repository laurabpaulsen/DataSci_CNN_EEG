import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# set font for all plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
from scipy.stats import binom


def chance_level(n, alpha = 0.001, p = 0.5):
    """
    Calculates the chance level for a given number of trials and alpha level

    Parameters
    ----------
    n : int
        The number of trials.
    alpha : float
        The alpha level.
    p : float
        The probability of a correct response.

    Returns
    -------
    chance_level : float
        The chance level.
    """
    k = binom.ppf(1-alpha, n, p)
    chance_level = k/n
    
    return chance_level

# create a color palette with 11 colors
colours = plt.cm.tab20(np.linspace(0, 1, 20))

def plot_results(results, output_dir):

    train_participants = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
    test_participants = ["sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11"]

    models = train_participants + ["joint", "ensemble", "finetuned_sub-06", "finetuned_sub-07", "finetuned_sub-08", "finetuned_sub-09", "finetuned_sub-10", "finetuned_sub-11"]
    participants = train_participants + test_participants

    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi = 300)

    for participant in participants:
        for model in models:
            key = f"{model}_{participant}"
            if key in results:
                accuracy = float(results[key]["accuracy"])*100

            if model == "joint" or model == "ensemble":
                if participant in train_participants:
                    marker = 'x'
            elif participant in model:
                marker = 'x'
            else:
                marker = 'o'

            ax.scatter(model, accuracy, marker=marker, color = colours[participants.index(participant)])


    # plot the chance level
    n = 69
    chance = chance_level(n, alpha = 0.05)*100
    ax.axhline(chance, color='black', linestyle='--', label="Chance level", linewidth=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")


    # turn the x labels 90 degrees
    ax.set_xticklabels(models, rotation=45)

    ax.set_ylim([25, 105])
    

    # make legend with the color of the participants
    handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colours[i], label=participant) for i, participant in enumerate(participants)]
    ax.legend(handles=handles, title="Participant", ncol=4, loc = "lower center")

    plt.tight_layout()

    plt.savefig(output_dir)
        


def plot_results_per_participant(results, output_dir):

    train_participants = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
    test_participants = ["sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11"]

    models = train_participants + ["joint", "ensemble", "finetuned_sub-06", "finetuned_sub-07", "finetuned_sub-08", "finetuned_sub-09", "finetuned_sub-10", "finetuned_sub-11"]
    participants = train_participants + test_participants

    
    fig, axes = plt.subplots(2, len(participants)//2, figsize=(10, 6), dpi = 300, sharey=True)

    for participant, ax in zip(participants, axes.flatten()):
        accuracies = []
        for model in models:
            key = f"{model}_{participant}"
            if key in results:
                accuracies.append(float(results[key]["accuracy"]))

            if model == "joint" or model == "ensemble":
                if participant in train_participants:
                    marker = 'o'
                else:
                    marker = 'x'
            else:
                marker = 'o'

        

        ax.scatter(models, accuracies, label=participant, marker=marker)
        ax.set_title(participant)

    # turn the x labels 90 degrees
    for ax in axes.flatten():
        ax.set_xticklabels(models, rotation=90)

    plt.tight_layout()

    plt.savefig(output_dir)


def plot_differences(results, output_dir):

    train_participants = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
    test_participants = ["sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11"]

    participants = train_participants + test_participants

    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi = 300)
    
    contrast_joint_ensemble = []
    contrast_joint_fine = []
    contrast_joint_individual = []
    
    for participant in participants:
            
        # CONTRASTING JOINT AND ENSEMBLE MODEL
        # get the differnce between the joint and the ensemble model
        joint_key = f"joint_{participant}"
        ensemble_key = f"ensemble_{participant}"

        accuracy_joint = float(results[joint_key]["accuracy"])*100
        accuracy_ensemble = float(results[ensemble_key]["accuracy"])*100

        difference = accuracy_joint - accuracy_ensemble
        contrast_joint_ensemble.append(difference)


        if participant in train_participants:
            marker = 'x'
        else:
            marker = 'o'

        ax.scatter(1, difference, marker=marker, color = colours[participants.index(participant)])

        # CONTRASTING JOINT AND FINE-TUNED MODELS FOR TEST PARTICIPANTS
        
        if participant in test_participants:

            joint_key = f"joint_{participant}"
            finetuned_key = f"finetuned_{participant}_{participant}"

            accuracy_joint = float(results[joint_key]["accuracy"])*100
            accuracy_finetuned = float(results[finetuned_key]["accuracy"])*100

            difference = accuracy_joint - accuracy_finetuned

            contrast_joint_fine.append(difference)

            ax.scatter(2, difference, marker="s", color = colours[participants.index(participant)])

       
        if participant in train_participants:
            joint_key = f"joint_{participant}"
            individual_key = f"{participant}_{participant}"

            accuracy_joint = float(results[joint_key]["accuracy"])*100
            accuracy_individual = float(results[individual_key]["accuracy"])*100

            difference = accuracy_joint - accuracy_individual
            
            contrast_joint_individual.append(difference)

            ax.scatter(3, difference, marker="s", color = colours[participants.index(participant)])
    
    # line at mean but only for the given label (e.g. joint - ensemble)

    mean_diff_joint_ensemble = np.mean(contrast_joint_ensemble)
    ax.axhline(mean_diff_joint_ensemble, color='red',xmin=0, xmax=1/3, linewidth=2)
    
    mean_diff_joint_fine = np.mean(contrast_joint_fine)
    ax.axhline(mean_diff_joint_fine, color='red', xmin=1/3, xmax=2/3, linewidth=2)

    mean_diff_joint_individual = np.mean(contrast_joint_individual)
    ax.axhline(mean_diff_joint_individual, color='red', xmin=2/3, xmax=1, linewidth=2)

    # x lim
    ax.set_xlim([0.5, 3.5])

    # only show ticks at 1, 2, 3 and change them to the contrast
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Joint - Ensemble", "Joint - Fine-tuned", "Joint - Individual"])
    

    ax.set_ylabel("Difference in accuracy (%)")


    # turn the x labels 90 degrees
    #ax.set_xticklabels(, rotation=45)

    # line at 0
    ax.axhline(0, color='black', linestyle='--', label="No difference", linewidth=0.5)


    # make legend with the color of the participants
    handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colours[i], label=participant) for i, participant in enumerate(participants)]
    ax.legend(handles=handles, title="Participant", ncol=4)

    plt.tight_layout()

    plt.savefig(output_dir)



def get_numbers(results):
    train_participants = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
    test_participants = ["sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11"]

    participants = train_participants + test_participants

    joint_test = []
    finetuned_test = []

    joint_train = []
    individual_train = []

    for participant in participants:
        
        if participant in test_participants:

            joint_key = f"joint_{participant}"
            finetuned_key = f"finetuned_{participant}_{participant}"

            accuracy_joint = float(results[joint_key]["accuracy"])*100
            accuracy_finetuned = float(results[finetuned_key]["accuracy"])*100

            joint_test.append(accuracy_joint)
            finetuned_test.append(accuracy_finetuned)

       
        if participant in train_participants:
            joint_key = f"joint_{participant}"
            individual_key = f"{participant}_{participant}"

            accuracy_joint = float(results[joint_key]["accuracy"])*100
            accuracy_individual = float(results[individual_key]["accuracy"])*100

            joint_train.append(accuracy_joint)
            individual_train.append(accuracy_individual)

    print(joint_test)
    
    print("Joint test:", np.mean(joint_test).round(2), np.std(joint_test).round(2))
    print("Finetuned test:", np.mean(finetuned_test).round(2), np.std(finetuned_test).round(2))
    print("Joint train:", np.mean(joint_train).round(2), np.std(joint_train).round(2))
    print("Individual train:", np.mean(individual_train).round(2), np.std(individual_train).round(2))


def create_table(results, output_path):

    models = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "joint", "ensemble", "finetuned_sub-06", "finetuned_sub-07", "finetuned_sub-08", "finetuned_sub-09", "finetuned_sub-10", "finetuned_sub-11"]
    results_for_table = {}

    for model in models:
        accuracies = []
        for key in results:
            if model in key:
                accuracies.append(float(results[key]["accuracy"])*100)
        
        results_for_table[model] = {"mean": np.mean(accuracies).round(2), "std": np.std(accuracies)}

    df = pd.DataFrame(results_for_table).T

    # output as markdown
    df.to_markdown(output_path)




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
    

    plot_results(results, output_dir / "accuracies.png")
    plot_results_per_participant(results, output_dir / "results.png" )
    plot_differences(results, output_dir / "differences.png")
    create_table(results, output_dir / "results.txt")

    get_numbers(results)
