import numpy as np
def postprocess(y):
    """ do postprocessing on predictions

    Args:
        y (list): predictions

    Returns:
        list: new predictions
    """
    res = []
    y = np.array(y)
    sz = y[0][0].shape[0]
    y = y.reshape((-1,6,sz,sz))
    for images in y:
        one = np.fliplr(images[1])
        two = np.rot90(images[2], k=3)
        three = np.rot90(images[3], k=2)
        four = np.rot90(images[4],k=1)
        five = np.flipud(images[5]) 
  
        m = np.stack([images[0], one, two, three, four, five])
        m = np.mean(m,axis=0)
        res.append(m)
    return res