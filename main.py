import argparse
import mxnet as mx
import cv2

from text_detection.text_detect import get_text_detector, text_detect
from text_recognition.text_reconition import text_recognize
from table_detection.table_detect import get_table_detector, table_detect
from serv_detection.serv_detect import serv_detect


def get_img_bbs(frame, text_bbs, extra):
    imgs = []
    for bbs in text_bbs:
        x = int(bbs[0])
        y = int(bbs[1])
        w = int(bbs[2] - bbs[0])
        h = int(bbs[3] - bbs[1])
        imgs.append(frame[y:y + h + extra, x:x + w + extra])
    return imgs


def main():
    parser = argparse.ArgumentParser(description="Tennis detection")
    parser.add_argument('--prefix-table', type=str, dest='prefix_table',
                        default='table_detection/resnet50/deploy_ssd_resnet50_512',
                        help='location of the table detection model')
    parser.add_argument('--epoch-table', dest='epoch_table', type=int, default=30,
                        help='epoch of the table detection model')
    parser.add_argument('--data-shape-table', dest='data_shape_table', type=int, default=512,
                        help='data shape table detection')
    parser.add_argument('--mean-pixels', type=str, dest='mean_pixels', default='104, 117, 123', help='mean pixels')
    parser.add_argument('--thresh', type=float, dest='thresh', default=0.5, help='threshold')

    parser.add_argument('--prefix-text', type=str, dest='prefix_text',
                        default='text_detection/model/frozen_east_text_detection.pb', help='text detection model')
    parser.add_argument('--data-shape-text', dest='data_shape_text', type=int, default=192,
                        help='data shape text detection model')

    parser.add_argument('--text-recognition-exe', type=str, dest='text_recognition_exe',
                        default='C:/Program Files/Tesseract-OCR/tesseract.exe', help='text recognition exe')
    parser.add_argument('--extra-text', dest='extra_text', type=int, default=10,
                        help='')

    parser.add_argument('--extra-serv', dest='extra_serv', type=int, default=15,
                        help='take extra pixels right for serv detection')

    parser.add_argument('--video-path', type=str, dest='video_path', default='video/short.mp4', help='video location')
    parser.add_argument('--gpu', type=int, dest='gpu', default=0, help='gpu index to use')
    args = parser.parse_args()

    video_path = args.video_path

    # Table detection model
    prefix_table_model = args.prefix_table
    epoch_table_model = args.epoch_table
    data_shape_table = args.data_shape_table
    mean_pixels = args.mean_pixels.split(',')
    mean_pixels = (int(mean_pixels[0]), int(mean_pixels[1]), int(mean_pixels[2]))
    thresh = args.thresh
    # get table detector
    table_detector = get_table_detector(prefix_table_model, int(epoch_table_model), data_shape_table,
                                        mean_pixels, mx.gpu(args.gpu))

    # Text detection model
    prefix_text_model = args.prefix_text
    data_shape_text = args.data_shape_text
    # Get text detector
    text_detector = get_text_detector(prefix_text_model)

    # Text recognition
    extra_text = args.extra_text
    tesseract_path = args.text_recognition_exe

    # Serv detection
    extra_right_serv = args.extra_serv

    cap = cv2.VideoCapture(video_path)
    while (True):
        print('######################################################################################################')
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (data_shape_table, data_shape_table))

        # Table detection
        table_bbs = table_detect(table_detector, frame, thresh, mean_pixels, data_shape_table)
        table_imgs = get_img_bbs(frame, table_bbs, 0)

        text_bbs_list = []
        text_bbs_img_list = []
        # Text detection
        for img in table_imgs:
            # input table image
            text_bbs = text_detect(img, text_detector, data_shape_text)
            text_bbs_list = text_bbs_list + text_bbs  # need for visualization
            text_bbs_img = get_img_bbs(img, text_bbs, extra_text)
            text_bbs_img_list = text_bbs_img_list + text_bbs_img

            # Serv detection
            player_serving = serv_detect(img, extra_right_serv)

        # Text recognition
        for img in text_bbs_img_list:
            player_names = text_recognize(img, tesseract_path)
            print('Player name: ', player_names)

        for bbs in table_bbs:
            cv2.rectangle(frame, (bbs[0], bbs[1]), (bbs[2], bbs[3]), (0, 0, 255), 2)
        try:
            text_detection_frame = table_imgs[0]
            for bbs in text_bbs_list:
                cv2.rectangle(text_detection_frame, (bbs[0], bbs[1]), (bbs[2], bbs[3]), (0, 255, 0), 2)
            text_detection_frame = cv2.resize(text_detection_frame, (300, 100))
            cv2.imshow('text_detection', text_detection_frame)
        except:
            print('')

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
