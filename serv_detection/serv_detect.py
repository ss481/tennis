import cv2
import numpy as np


def extractHistogram(frame):
    """
    Extract histogram for every chanel of the input frame
    Args:
        frame: cv2

    Returns:
        hist_blue, hist_green, hist_red
    """
    numPixels = np.prod(frame.shape[:2])
    (b, g, r) = cv2.split(frame)
    histb = cv2.calcHist([b], [0], None, [16], [0, 256]) / numPixels
    histg = cv2.calcHist([g], [0], None, [16], [0, 256]) / numPixels
    histr = cv2.calcHist([r], [0], None, [16], [0, 256]) / numPixels

    n_histb = cv2.normalize(histb, histb).flatten()
    n_histg = cv2.normalize(histg, histg).flatten()
    n_histr = cv2.normalize(histr, histr).flatten()

    return n_histb, n_histg, n_histr


def serv_detect(img, extra_right):
    """
    Input image is the output of the table detection model

    Args:
        img: cv2
        extra_right: int

    Returns:
        name of the player who is serving: str
    """
    (h, w) = img.shape[:2]
    ser_img = img[0:h, 0:extra_right]
    cv2.imshow('serv', ser_img)
    player1_ser = ser_img[0:int(h / 2), 0:extra_right]
    player2_ser = ser_img[int(h / 2):h, 0:extra_right]
    #n_histb1, n_histg1, n_histr1 = extractHistogram(player1_ser)
    #n_histb2, n_histg2, n_histr2 = extractHistogram(player2_ser)

    player1_ser_gray = cv2.cvtColor(player1_ser, cv2.COLOR_BGR2GRAY)
    player2_ser_gray = cv2.cvtColor(player2_ser, cv2.COLOR_BGR2GRAY)
    player1_gray_hist = cv2.calcHist([player1_ser_gray], [0], None, [256], [0, 256])
    player2_gray_hist = cv2.calcHist([player2_ser_gray], [0], None, [256], [0, 256])

    if np.max(player1_gray_hist) < np.max(player2_gray_hist):
        print('Player1 is serving')
        return 'player1'
    else:
        print('Player2 is serving')
        return 'player2'
