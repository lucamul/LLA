from parameters import SEED, K_FOLD, DEVICE, BATCH_SIZE, LR, TH
from dataset import RoadSegmentationDataset
from model import RoadSegmentationModel
from torch.utils.data import DataLoader
from helpers import split_data
import numpy as np

def build_k_indices(N, k_fold, seed = SEED):
    """build k indices for k-fold.

    Args:
        N:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = N
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_step(data, gts, k_indices, k, th, lr):
    """ Performs one step of cross validation

    Args:
        data (np.array): images
        gts (np.array): ground truths
        k_indices (np.array): Indices to split data in test and train
        k (int): Index for test
        th (float): Foreground threshold
        lr (float): Learning rate

    Returns:
       float : f1 score of this step
    """
    train_data = np.delete(data, k_indices[k], axis = 0)
    train_gts = np.delete(gts, k_indices[k], axis = 0)
    test_x = []
    test_y = []
    for id in k_indices[k]:
        test_x.append(data[id])
        test_y.append(gts[id])
    operations = {}
    operations['augment'] = True
    operations['normalization'] = True
    operations['patches'] = True

    
    train_set = RoadSegmentationDataset(train_data,train_gts,operations, True, DEVICE)
    test_set = RoadSegmentationDataset(test_x,test_y,operations,True,DEVICE)
    evaluation_dataset = RoadSegmentationDataset(test_x,test_y,operations,False,DEVICE)

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set,BATCH_SIZE, shuffle=False)
    evaluation_loader = DataLoader(evaluation_dataset,1, shuffle=False)

    model = RoadSegmentationModel(DEVICE,th=th,lr=lr,max_iter=10)
    results = model.train(train_loader, test_loader,True,evaluation_loader,False)
    f1_score = results["f1"]
    return f1_score


def validation_over_threshold(data, gts, k_indices, k, threshold):
    """ Performs CV-step over a threshold

    Args:
        data (np.array): Images
        gts (np.array): Ground Truths
        k_indices (np.array): Indices to split test and train set
        k (int): Index for train
        threshold (float): foreground threshold

    Returns:
        float : f1 score
    """
    return cross_validation_step(data,gts,k_indices,k,threshold,LR)

def validation_over_learning_rate(data, gts, k_indices, k, lr):
    """Performs CV-step over a learning rate

    Args:
        data (np.array): Images
        gts (np.array): Ground Truths
        k_indices (np.array): Indices to split test and train set
        k (int): Index for train
        lr (float): learning rate

    Returns:
        float : f1 score
    """
    return cross_validation_step(data,gts,k_indices,k,TH,lr)

def cross_validation(data, gts, parameters, parameter_name, N, seed = SEED, k_fold = K_FOLD):
    """ Performs cross validation over parameter_name trying all parameters

    Args:
        data (np.array): Images
        gts (np.array): Ground Truths
        parameters (array): list of parameters
        parameter_name (str): either threshold or learning rate
        N (int): data size
        seed (int, optional): seed for randomness. Defaults to SEED.
        k_fold (int, optional): number of k folds. Defaults to K_FOLD.

    Returns:
        (float, float): optimal parameter and f1 score
    """
    k_indices = build_k_indices(N,k_fold,seed)
    best_performance = -1
    optimal_parameter = -1

    for parameter in parameters:
        print("Trying " + str(parameter_name) + " = " + str(parameter))
        avg_performance = 0
        performances = np.zeros(k_fold)

        for k in range(k_fold):
            performance = 0
            if parameter_name == "threshold":
                performance = validation_over_threshold(data,gts,k_indices,k,parameter)[-1]
            elif parameter_name == "learning rate":
                performance = validation_over_learning_rate(data,gts,k_indices,k,parameter)[-1]
            avg_performance = performance + avg_performance
            performances[k] = performance
        
        avg_performance = avg_performance / k_fold

        print("Cross-Validation for " + parameter_name + " = " + str(parameter) + " with f1_score = " + str(avg_performance))
        if best_performance == -1 or avg_performance > best_performance:
            best_performance = avg_performance
            optimal_parameter = parameter

    print("Optimal Patameter for " + parameter_name + " = " + str(optimal_parameter))
    return optimal_parameter, best_performance