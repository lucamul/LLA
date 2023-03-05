import json
from glob import glob
import numpy as np
from PIL import ImageFile
import sys, getopt
import torch
from torch.utils.data import DataLoader
sys.path.append('..')

from model import RoadSegmentationModel
from dataset import RoadSegmentationDataset
from helpers import split_data
from parameters import *
from cross_validation import *

normalize = False
divide_patches = False
augment = False
training = True
submmission = False
validation = False

def main(argv):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    opts, args = getopt.getopt(argv, 'v', ["experiment=","epochs="])
    
    global normalize
    global divide_patches
    global augment
    global training
    global submmission
    global validation
    for opt, arg in opts:
        if opt == "-v":
            validation = True
        elif opt == "--experiment":
            if int(arg) == 1:
                divide_patches = True
                normalize = False
                augment = False
                training = True
                submmission = True
            elif int(arg) == 2:
                divide_patches = True
                normalize = True
                augment = False
                training = True
                submmission = True
            elif int(arg) == 3:
                divide_patches = True
                normalize = False
                augment = True
                training = True
                submmission = True
            elif int(arg) == 4:
                divide_patches = True
                normalize = False
                augment = True
                training = True
                submmission = True
                global TH
                global LR
                TH = 0.25
                LR = 0.0003
        elif opt == "--epochs":
            global MAX_ITER
            MAX_ITER = int(arg)
        else:
            print("ERROR: this option is not allowed: " + opt)
    
    operations = {}
    operations['augment'] = augment
    operations['normalization'] = normalize
    operations['patches'] = divide_patches

    data = sorted(glob(DATA_PATH + "/training/images/*.png"))
    gts = sorted(glob(DATA_PATH + "/training/groundtruth/*.png"))

    optimal_th = TH
    optimal_lr = LR

    if validation:
        print("Starting Cross Validation over Foreground Threshold")
        optimal_th,_ = cross_validation(data,gts,THRESHOLD_VALIDATION_VECTOR,"threshold",len(data))
        print("Starting Cross Validation over Learning Rate")
        optimal_lr,_ = cross_validation(data,gts,LEARNING_RATE_VALIDATION_VECTOR,"learning rate",len(data))

    train_data, train_labels, test_data, test_labels = split_data(data,gts)
    if submmission:
        train_data = data
        train_labels = gts
    train_set = RoadSegmentationDataset(train_data,train_labels,operations, True, DEVICE)
    test_set = RoadSegmentationDataset(test_data,test_labels,operations,True,DEVICE)

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set,BATCH_SIZE, shuffle=False)

    model = RoadSegmentationModel(DEVICE,th = optimal_th, lr = optimal_lr)

    if training:
        results = model.train(train_loader, test_loader,False,None,augment)
        loss = results['train_loss']
        f1 = results['f1']
        accuracy = results['accuracy']
        test_loss = results['test_loss']
        print("TRAINING LOSS = " + str(loss[len(loss)-1]))
        print("TEST LOSS = " + str(test_loss[len(test_loss)-1]))
    
    if submmission:
        submission_images = sorted(glob(DATA_PATH + "/test_set_images/*/*"),
            key = lambda x: int(x.split('/')[-2].split('_')[-1]))

        test_set = RoadSegmentationDataset(submission_images, None, operations, False, DEVICE)
        test_loader = DataLoader(test_set,1)

        model.submit(test_loader)

if __name__  == "__main__":
    main(sys.argv[1:])