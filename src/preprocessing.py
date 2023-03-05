import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from helpers import images_to_np_array, PIL_Images_from_np_array
from parameters import *

def preprocess(data, gts, operations, train: bool):
    """ do pre processing

    Args:
        data (list): images
        gts (list): ground truths
        operations (dict): what pre processing operations to do
        train (bool): is this training or testing?

    Returns:
        (list, list) : data and groundtruths after pre processing
    """
    data = images_to_np_array(data)
    if train:
        gts = images_to_np_array(gts)
    else:
        gts = None
    
    if operations['normalization']:
        for i, image in enumerate(data):
            image = image / 255
            image = ((image - image.mean(axis=(0,1),dtype='float64')))/(image.std(axis=(0,1),dtype='float64'))
            data[i] = image

    if not train:
        return data, None
    
    if operations['augment']:
        data = PIL_Images_from_np_array(data)
        if gts is not None:
            gts = PIL_Images_from_np_array(gts)

        augmented_images = []
        augmented_groundtruths = []

        for image in data:
            augmented_imgs = []
            augmented_imgs.append(image.transpose(Image.FLIP_LEFT_RIGHT))

            augmented_imgs.append(image.transpose(Image.ROTATE_90))

            augmented_imgs.append(image.transpose(Image.ROTATE_180))

            augmented_imgs.append(image.transpose(Image.ROTATE_270))

            augmented_imgs.append(image.transpose(Image.FLIP_TOP_BOTTOM))
            
            
            if train:
              augmented_imgs.append(image.filter(ImageFilter.GaussianBlur(4)))
              color_shift = ImageEnhance.Color(image)
              augmented_imgs.append(color_shift.enhance(0.5))
              augmented_imgs.append(image.rotate(10, resample=Image.BICUBIC))
              augmented_imgs.append(image.rotate(20, resample=Image.BICUBIC))
              augmented_imgs.append(image.rotate(30, resample=Image.BICUBIC))
              augmented_imgs.append(image.rotate(40, resample=Image.BICUBIC))
              augmented_imgs.append(image.rotate(50, resample=Image.BICUBIC))
              augmented_imgs.append(image.rotate(60, resample=Image.BICUBIC))
 
            augmented_images.extend(augmented_imgs)
        if gts is not None:
            for image in gts:
                augmented_imgs = []
                augmented_imgs.append(image.transpose(Image.FLIP_LEFT_RIGHT))

                augmented_imgs.append(image.transpose(Image.ROTATE_90))

                augmented_imgs.append(image.transpose(Image.ROTATE_180))

                augmented_imgs.append(image.transpose(Image.ROTATE_270))

                augmented_imgs.append(image.transpose(Image.FLIP_TOP_BOTTOM))

                if train:
                  augmented_imgs.append(image)
                  augmented_imgs.append(image)
                  augmented_imgs.append(image.rotate(10, resample=Image.BICUBIC))
                  augmented_imgs.append(image.rotate(20, resample=Image.BICUBIC))
                  augmented_imgs.append(image.rotate(30, resample=Image.BICUBIC))
                  augmented_imgs.append(image.rotate(40, resample=Image.BICUBIC))
                  augmented_imgs.append(image.rotate(50, resample=Image.BICUBIC))
                  augmented_imgs.append(image.rotate(60, resample=Image.BICUBIC))
                
                augmented_groundtruths.extend(augmented_imgs)
        data.extend(augmented_images)
        if gts is not None:
            gts.extend(augmented_groundtruths)
        data = np.array([np.array(image) for image in data])
        if gts is not None:
            gts = np.array([np.array(image) for image in gts])

    if operations['patches'] and gts is not None:
        patch_size = PATCH_SIZE
        data = [crop(image,patch_size,patch_size) for image in data]
        data = np.asarray([data[i][j] for i in range(len(data)) for j in range(len(data[i]))])
        gts = [crop(image,patch_size,patch_size) for image in gts]
        gts = np.asarray([gts[i][j] for i in range(len(gts)) for j in range(len(gts[i]))])
        data = images_to_np_array(data)
        gts = images_to_np_array(gts)

    return data, gts
        
def crop(image, width, height):
    """ crop an image

    Args:
        image(list): image to crop
        width (int): width to crop to
        height (int): height to crop to

    Returns:
        list: new image
    """
    res = []
    for i in range(0,image.shape[1],height):
        for j in range(0,image.shape[0],width):
            if len(image.shape) == 2:
                res.append(image[j:j + width, i : i + height])
            else:
                res.append(image[j:j + width, i : i + height, :])
    return res

def load_data(path_data, path_gts, train : bool, device, operations):
    """ load data from path

    Args:
        path_data (str): path for data
        path_gts (str): path for ground truths
        train (bool): is it train data?
        device (str): device to use
        operations (dict): operations to do in preprocessing

    Returns:
        _type_: _description_
    """
    data = [Image.open(img) for img in path_data]
    gts = []
    if path_gts is not None:
        gts = [Image.open(gt) for gt in path_gts]
    
    if train:
        data, gts = preprocess(data, gts, operations,True)
    else:
        data, _ = preprocess(data, None, operations, False)

    return data, gts