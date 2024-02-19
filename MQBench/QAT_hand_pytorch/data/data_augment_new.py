import cv2
import numpy as np
import random



def resize_pad(img, target_w, target_h):
    """ resize and pad images to be input to the detectors

    The face and palm detector networks take 192x192 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img1: 192x192
        scale: scale factor between original image and 256x256 image
        pad: pixels of padding in the original image
    """

    size0 = img.shape
    if size0[0]>=size0[1]:
        h1 = target_h
        w1 = target_w * size0[1] // size0[0]
        padh = 0
        padw = target_w - w1
        scale = size0[1] / w1
    else:
        h1 = target_h * size0[0] // size0[1]
        w1 = target_w
        padh = target_h - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh%2
    padw1 = padw//2
    padw2 = padw//2 + padw%2
    img1 = cv2.resize(img, (w1,h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)),'constant')

    pad = (int(padh1), int(padw1))
    return img1, scale, pad


def resize_landmarks(landmarks, scale, pad):

    landmarks[:, 0] = landmarks[:, 0] / scale + pad[1]
    landmarks[:, 1] = landmarks[:, 1] / scale + pad[0]
    landmarks[:, 2] = landmarks[:, 2] / scale + pad[1]
    landmarks[:, 3] = landmarks[:, 3] / scale + pad[0]

    landmarks[:, 4::2] = landmarks[:, 4::2] / scale + pad[1]
    landmarks[:, 5::2] = landmarks[:, 5::2] / scale + pad[0]

    return landmarks


def draw_landmarks(img, detections, with_keypoints=True):
    # if isinstance(detections, torch.Tensor):
    #     detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    n_keypoints = detections.shape[1] // 2 - 2

    for i in range(detections.shape[0]):
        xmin = detections[i, 0]
        ymin = detections[i, 1]
        xmax = detections[i, 2]
        ymax = detections[i, 3]

        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1)

        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2])
                kp_y = int(detections[i, 4 + k*2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)
                cv2.imshow("hdetect", img)
                cv2.waitKey(1000)
    return img