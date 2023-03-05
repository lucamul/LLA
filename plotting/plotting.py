import matplotlib.pyplot as plt
import numpy as np
import csv
from plot_parameters import *

def plot(x,y,x_label,y_label, metric, title):
    """ plot the given data

    Args:
        x (np.array): xs
        y (np.array): ys
        x_label (str): label
        y_label (str): label
        metric (str): which metric to plot
        title (str): plot title
    """
    fig, ax = plt.subplots()
    plt.ticklabel_format(axis='y', style='plain', useMathText=True, scilimits=(0,0))
    #plt.tight_layout()
    ax.set_ylabel(y_label, fontsize=AXES_FONT)
    ax.set_xlabel(x_label, fontsize=AXES_FONT)
    ax.set_title(title, fontdict = TITLE_DICT)
    if GRID:
        ax.grid(GRID,color='gray',linestyle = '--', linewidth = 1, axis = 'y')

    ax.tick_params(axis='x',labelsize=TICK_FONT)
    ax.tick_params(axis='y',labelsize=TICK_FONT)
    if metric == "experiments":
        ax.bar(x,y, width = 0.4,color = COLORS[metric])
    else:
        ax.plot(x,y, marker = MARKERS[metric], color = COLORS[metric], linewidth=LINE_WIDTH, markersize = MARKER_SIZE, markeredgecolor=COLORS[metric], markerfacecolor='None')
    plt.savefig("./" + title + ".pdf")

def plot_vs_num_epochs(path = ""):
    """ plot f1 score vs number of epochs

    Args:
        path (str, optional): path for CSV data. Defaults to "".
    """
    if path == "":
        print("provide a valid path")
        return
    f1_scores = []
    num_epochs = []
    accs = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file,delimiter=",")
        for i,row in enumerate(reader):
            if i == 0:
                continue
            
            num_epochs.append(row[0])
            f1_scores.append(float(row[1]))
            accs.append(float(row[2]))
    plot(num_epochs,f1_scores,"number of training epochs","f1 score","num_epochs","number of epochs vs f1 score")
    plot(num_epochs,accs,"number of training epochs","accuracy","num_epochs","number of epochs vs accuracy")

def plot_vs_expertiment(path = ""):
    """ plot f1 score bar graph for experiments

    Args:
        path (str, optional): path for CSV data. Defaults to "".
    """
    if path == "":
        print("provide a valid path")
        return
    f1_scores = []
    accs = []
    experiments = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file,delimiter=",")
        for i,row in enumerate(reader):
            if i == 0:
                continue
            
            experiments.append(row[0])
            f1_scores.append(float(row[1]))
            accs.append(float(row[2]))

    plot(experiments,f1_scores,"experiments","f1 scores","experiments","f1 scores for different experiments")
    plot(experiments,accs,"experiments","accuracies","experiments","accuracies for different experiments")

if __name__ == "__main__":
    plot_vs_num_epochs(path="f1_numepochs.csv")
    plot_vs_expertiment(path="f1_experiments.csv")