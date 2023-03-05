import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from parameters import *

def image_to_tensor(image:np.ndarray, device = None, divide = False):
    """ transform an image to tensor

    Args:
        image (np.ndarray): Image
        device (str, optional): device to use. Defaults to None.
        divide (bool, optional): whether to divide. Defaults to False.

    Returns:
        torch.Tensor: tensor version of the image
    """
    image = np.transpose(image,[2,0,1])
    tensor = torch.Tensor(image)
    if divide:
        tensor = tensor / 255
    tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def mask_to_tensor(mask: np.ndarray, device = None):
    """ transforms a mask to a tensor

    Args:
        mask (np.ndarray): mask to transform
        device (str, optional): device to use. Defaults to None.

    Returns:
        torch.Tensor : tensor version of mask
    """
    tensor = transforms.ToTensor()(mask)
    tensor = torch.round(tensor)
    if device is not None:
        tensor = tensor.to(device)
    return tensor[0,:,:][None,:,:]

def transform_to_patch(pixels, th):
    m = np.mean(pixels)
    if m  > th:
        return 1
    else:
        return 0


def transform_prediction_to_patch(img, id, patch_size=16,step=16,th=0.25):
    """ gets prediction for an image and translates it to prediction on patches

    Args:
        img (np.array): image
        id (int): id of the imahe
        patch_size (int, optional): size of a patch. Defaults to 16.
        step (int, optional): step size. Defaults to 16.
        th (float, optional): foreground threshold. Defaults to 0.25.

    Returns:
        (list, list): predictions and respective ids
    """
    prs = []
    ids = []
    for j in range(0,img.shape[1],step):
        for i in range(0, img.shape[0],step):
            prs.append(transform_to_patch(img[i:i+patch_size,j:j+patch_size],th=th))
            ids.append("{:03d}_{}_{}".format(id, j, i))
    return prs, ids

def images_to_np_array(images):
    """ transforms images to numpy array

    Args:
        images (list): list of images

    Returns:
        np.array : images as numpy array
    """
    res = []
    for img in images:
        img = np.array(img)
        res.append(img)
    return np.array(res)

def PIL_Images_from_np_array(images):
    """ generates list of PIL images from numpy array

    Args:
        images (np.array): list of images

    Returns:
        list(Image) : list of PIL images
    """
    res = []
    for img in images:
        res.append(Image.fromarray(img))
    return res

def split_data(x,y):
    """  Splits data into two sets (train and test)

    Args:
        x (np.array): images
        y (np.array): ground truths

    Returns:
        (np.array,np.array,np.array,np.array): train x and ys and test x and ys
    """
    np.random.seed(SEED)
    ids = np.arange(len(x))

    np.random.shuffle(ids)

    division = len(ids)*(1-TEST_SIZE)
    division = int(division)
    return np.array(x)[ids[:division]], np.array(y)[ids[:division]], np.array(x)[ids[division:]], np.array(y)[ids[division:]]