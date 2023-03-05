# Machine Learning Course Project 2

This codebase is the implementation of the Road Segmentation Challenge of the lla_team:

- **Luca Multazzu**
- **Lorenzo Brusca**
- **Aline Janvier**

## Codebase description

* `run notebook`

Jupyter Notebook to use on Google Colab

`src`:

* `network`

Has the deeplab pre-trained model we use as base

* `cross_validation`

Contains the scripts for cross validation over foreground threshold and learning rate 

* `dataset`

Contains the class of the Road Segmentation dataset

* `helpers`

Contains a number of helper functions used throughout the codebase

* `model`

This class has the model, with the training, test and submit methods

* `parameters`

Has the parameters of the run

* `post processing`

A small script for post processing

* `pre processing`

Method that perform pre processing on the data

* `run`

This script actually runs the code

`plotting`:

* `plot_parameters`

Parameters for the plot

* `plotting`

Plotting scripts

Also includes CSV files of the scores obtained via AI crowd

## Set-Up to run on personal machine

after cloning the repository create the virtual environment with:

`python3 -m pip install --user virtualenv`

`virtualenv -p python3 MLProject2`

`source MLProject2/bin/activate`

`python3 -m pip install -r requirements.txt`

## Set-Up to run on Google Colab

- Open the run.ipynb notebook to Colab
- Add the requirements.txt file to Colab
- Load the data folders to your Google Drive
- Change the data paths in the notebook to the ones in your drive
- Install requirements, then restart runtime

## Run on personal machine

Run with the command:
`cd src`
`python3 run.py`

There are four experiments we ran for the report, to choose which experiment to run use option `--experiment=` followed by number 1, 2. 3 or 4:

- Experiment 1 runs default configurations
- Experiment 2 runs with normalization on top of 1
- Experiment 3 further adds data augmentation but removes normalization
- Experiment 4 uses Learning Rate and Foreground Threshold found via cross-validation

You can also run the cross validation yourself with the `-v` option. (Note this takes a lot of time)

You can select the number of epochs to run for with option `--epochs=` (Default is 64)

## Run on colab

After following the instructions to set up, you can choose which experiment to run in the EXPERIMENT constant (either 1, 2, 3 or 4)

You can also run cross validation by setting validation=True in the configuration cell

You can change number of epochs by changing MAX_ITER in the parameters cell

Then run all the cells (Except the requirement cell and the mount drive cell which have already been run).