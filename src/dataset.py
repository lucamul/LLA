import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from helpers import mask_to_tensor, image_to_tensor
from preprocessing import load_data

class RoadSegmentationDataset(Dataset):
    def __init__(self, data_path, gt_path, operations: dict, train: bool, device = None):
        """ initialize dataset

        Args:
            data_path (str): where to fetch images from
            gt_path (str): where to fetch ground truths from
            operations (dict): what operations to do in pre-processing
            train (bool): whether this is used for training
            device (str, optional): which device are you using. Defaults to None.
        """
        self.train = train
        self.device = device
        imgs, gts = load_data(data_path,gt_path,train,device,operations)
        divide = not operations['normalization']
        if gts is not None:
            self.gt = [mask_to_tensor(gt,device) for gt in gts]
        self.data = [image_to_tensor(img, device, divide = divide) for img in imgs] 

    def __len__(self):
        """ get dataset length

        Returns:
            int: dataset length
        """
        return len(self.data)

    def __getitem__(self, index):
        """ get item at a specific index

        Args:
            index (int): index

        Returns:
            (list[Tensor],list[Tensor]): data and ground truth at index
        """
        if self.train:
            return self.data[index], self.gt[index]
        else:
            return self.data[index]