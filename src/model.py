import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from pretrained_network.network import deeplab_model
from postprocessing import postprocess
from helpers import transform_prediction_to_patch
from parameters import *
from tqdm import tqdm

class RoadSegmentationModel(nn.Module):
    def __init__(self, device, lr = LR, th =TH, max_iter = MAX_ITER):
        """ initialize model

        Args:
            device (str): _device to use
            lr (float, optional): learning rate. Defaults to LR.
            th (float, optional): foreground threshold. Defaults to TH.
            max_iter (int, optional): how many epochs to train for. Defaults to MAX_ITER.
        """
        super().__init__()
        self.device = device
        self.pre_trained_network = deeplab_model()
        self.criterion = nn.BCEWithLogitsLoss()
        self.pre_trained_network.to(self.device)
        self.th = th
        self.lr = lr
        self.max_iter = max_iter

    def forward(self, data):
        data_x = data[0].to(self.device)
        return self.pre_trained_network(data_x)

    def train_epoch(self, loader, optimizer):
        """ do one train epoch

        Args:
            loader (torch.utils.data.DataLoader): loader for the data
            optimizer (torch.optim): optimizer to use (Adam)

        Returns:
            list(float): loss for the epoch
        """
        loss = []
        self.pre_trained_network.train()
        for batch in tqdm(loader):
            pr = self.forward(batch)
            y = batch[1].to(self.device)
            l = self.criterion(pr,y).to(self.device)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l = l.cpu().detach().numpy()
            loss.append(l)
        return np.mean(loss)

    def test_epoch(self, loader):
        """ do one test epoch

        Args:
            loader (torch.utils.data.DataLoader): data loader for the test

        Returns:
            list(loss): test loss for epoch
        """
        self.pre_trained_network.eval()
        loss = []
        with torch.no_grad():
            for batch in tqdm(loader):
                pr = self.forward(batch)
                y = batch[1].to(self.device)
                l = self.criterion(pr,y).to(self.device)
                l = l.cpu().detach().numpy()
                loss.append(l)
        return np.mean(loss)

    def get_score(self,loader, do_postprocessing=True):
        """ evaluate one epoch

        Args:
            loader (torch.utils.data.DataLoader):  data loader
            do_postprocessing (bool, optional): whether to do the postprocessing. Defaults to True.

        Returns:
            (float,float): f1 score and accuracy
        """
        self.pre_trained_network.eval()
        prs = []
        ys = []
        first_predictions = self.make_prediction(loader,do_postprocessing)
        masks = loader.dataset.gt

        for pr, y in zip(first_predictions, masks):
            labels, _ = transform_prediction_to_patch(pr,1,th=self.th)
            prs.extend(labels)
            y = y[0].cpu().detach().numpy()
            real_labels, _ = transform_prediction_to_patch(y,1,th=self.th)
            ys.extend(real_labels)
        return f1_score(ys,prs), accuracy_score(ys,prs)

    def make_prediction(self, loader, do_postprocessing):
        """ perform a prediction

        Args:
            loader (torch.utils.data.DataLoader): data loader
            do_postprocessing (bool): whether to do postprocessing

        Returns:
            list(list(Tensor)): predictions
        """
        pr = []
        self.pre_trained_network.eval()
        sigmoid = torch.nn.Sigmoid()
        with torch.no_grad():
            for batch in tqdm(loader):
                p = sigmoid(self.pre_trained_network(batch.to(self.device))).cpu().detach().numpy()
                pr.append(p[0][0])
        if do_postprocessing:
            pr = postprocess(pr)
        return pr


    def train(self, train_loader, test_loader, evaluate, evaluate_loader, do_postprocessing):
        """ train the model

        Args:
            train_loader (torch.utils.data.DataLoader): Data loader for training
            test_loader (torch.utils.data.DataLoader): Data loader for testing 
            evaluate (bool): whether to evaluate
            evaluate_loader (torch.utils.data.DataLoader): Data loader for evaluation
            do_postprocessing (bool): whether to do postprocessing

        Returns:
            dict: dictionary containing the final metrics (train loss, f1 score, accuracy, test loss)
        """
        optimizer = torch.optim.Adam(self.pre_trained_network.parameters(), lr=self.lr)
        losses = []
        test_losses = []
        accuracies = []
        f1s = []
        best_loss = {'loss': float('inf'), 'epoch': 0}
        i = 1
        print("Starting Training")
        while True:
            print("EPOCH " + str(i))
            l_train = self.train_epoch(train_loader, optimizer)
            print("epoch trained, now testing")
            losses.append(l_train)
            l_test = self.test_epoch(test_loader)
            test_losses.append(l_test)
            print("Test Loss for epoch " + str(i) + " = " + str(l_test))
            f1 = 0
            acc = 0
            if evaluate:
                f1, acc = self.get_score(evaluate_loader, do_postprocessing)
                print("LOSS = " + str(l_test) + " F1 = " + str(f1) + " ACCURACY = " + str(acc))

            f1s.append(f1)
            accuracies.append(acc)

            if l_test < best_loss['loss']:
                best_loss['loss'] = l_test
                best_loss['epoch'] = i

            if i == self.max_iter - 1:
                break

            i += 1
        results = {}
        results['train_loss'] = losses
        results['f1'] = f1s
        results['accuracy'] = accuracies
        results['test_loss'] = test_losses
        return results

    def submit(self, test_loader):
        """ generate a submission for the AI crowd challenge

        Args:
            test_loader (torch.utils.data.DataLoader): Data Loader for the data
        """
        prs = self.make_prediction(test_loader,False)
        img_ids = range(1,len(prs)+1)
        
        ret_ids = []
        ret_labels = []

        for pr, i in zip(prs,img_ids):
            labels, ids = transform_prediction_to_patch(pr,i,th=self.th)
            for label in labels:
                ret_labels.append(label)
            for id in ids:
                ret_ids.append(id)
        
        pd.DataFrame({'id': ret_ids, 'prediction' : ret_labels}).to_csv(SUBMISSION_PATH,index=False)