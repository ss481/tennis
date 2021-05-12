import mxnet as mx
import numpy as np
import cv2
from detect.detector import Detector


def get_table_detector(prefix, epoch, data_shape, mean_pixels, ctx):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    """
    detector = Detector(None, prefix, epoch, data_shape, mean_pixels, ctx=ctx)
    print('[INFO] Table detection model loaded')

    return detector


def img_preprocessing(img, data_shape, mean_pixels):
    """
    Preprocess image to use for the deep learning model

    Args:
        img: cv2
        data_shape: int
        mean_pixels: tuple

    Returns:
        processed image : cv2
    """
    img = cv2.resize(img, (data_shape, data_shape))
    img = np.transpose(img, (2, 0, 1))
    img = img - np.reshape(mean_pixels, (3, 1, 1))
    return [mx.nd.array([img])]


def table_detect(detector, image, thresh, mean_pixels, data_shape):
    """
    Run the DL model for table detection

    Args:
        detector: mxnet
        image: cv2
        thresh: float
        mean_pixels: tuple
        data_shape: int

    Returns:
        bounding boxes of the table detection: list of bbs
    """
    e1 = cv2.getTickCount()
    data = img_preprocessing(image, data_shape, mean_pixels)
    det_batch = mx.io.DataBatch(data, [])
    detector.mod.forward(det_batch, is_train=False)
    dets = detector.mod.get_outputs()[0].asnumpy()[0, :, :]
    dets = dets[np.where(dets[:, 0] >= 0)[0]]
    bbs = []
    for i in range(dets.shape[0]):
        cls_id = int(dets[i][0])
        score = dets[i][1]
        if cls_id >= 0 and score >= thresh:
            xmin = int(dets[i, 2] * data_shape)
            ymin = int(dets[i, 3] * data_shape)
            xmax = int(dets[i, 4] * data_shape)
            ymax = int(dets[i, 5] * data_shape)
            bbs.append((xmin, ymin, xmax, ymax))
    e2 = cv2.getTickCount()
    t = (e2 - e1) / cv2.getTickFrequency()
    print('[INFO] Table detection: ', t)
    return bbs
