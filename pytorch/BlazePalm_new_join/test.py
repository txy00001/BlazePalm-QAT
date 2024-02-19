import numpy as np
import torch
import cv2
import glob
import os
import os.path
from models.blaze_palml_new import BlazePalm
import numpy as np

from models_QAT.QAT_net_pytorch import BlazePalm_QAT

SSD_ANCHOR_OPTIONS = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 192,
    "input_size_width": 192,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) * 0.5
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


def generate_anchors(options: dict):
    strides_size = len(options["strides"])
    assert options["num_layers"] == strides_size
    anchors = []
    layer_id = 0
    while layer_id < strides_size:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []

        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < strides_size) and \
                (options["strides"][last_same_stride_layer] == options["strides"][layer_id]):
            scale = calculate_scale(options["min_scale"],
                                    options["max_scale"],
                                    last_same_stride_layer,
                                    strides_size)

            if last_same_stride_layer == 0 and options["reduce_boxes_in_lowest_layer"]:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios.append(1.0)
                aspect_ratios.append(2.0)
                aspect_ratios.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)
            else:
                for aspect_ratio in options["aspect_ratios"]:
                    aspect_ratios.append(aspect_ratio)
                    scales.append(scale)

                if options["interpolated_scale_aspect_ratio"] > 0.0:
                    scale_next = 1.0 if last_same_stride_layer == strides_size - 1 \
                        else calculate_scale(options["min_scale"],
                                             options["max_scale"],
                                             last_same_stride_layer + 1,
                                             strides_size)
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(options["interpolated_scale_aspect_ratio"])

            last_same_stride_layer += 1
        print("len(aspect_ratios):",len(aspect_ratios))
        for i in range(len(aspect_ratios)):
            ratio_sqrts = np.sqrt(aspect_ratios[i])
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options["strides"][layer_id]
        feature_map_height = int(np.ceil(options["input_size_height"] / stride))
        feature_map_width = int(np.ceil(options["input_size_width"] / stride))

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options["anchor_offset_x"]) / feature_map_width
                    y_center = (y + options["anchor_offset_y"]) / feature_map_height

                    new_anchor = [x_center, y_center, 0, 0]
                    if options["fixed_anchor_size"]:
                        new_anchor[2] = 1.0
                        new_anchor[3] = 1.0
                    else:
                        new_anchor[2] = anchor_width[anchor_id]
                        new_anchor[3] = anchor_height[anchor_id]
                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer
    return anchors


def resize_pad(img):
    """ resize and pad images to be input to the detectors
    """

    size0 = img.shape
    if size0[0]>=size0[1]:
        h1 = 192
        w1 = 192 * size0[1] // size0[0]
        padh = 0
        padw = 192 - w1
        scale = size0[1] / w1
    else:
        h1 = 192 * size0[0] // size0[1]
        w1 = 192
        padh = 192 - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh%2
    padw1 = padw//2
    padw2 = padw//2 + padw%2
    img1 = cv2.resize(img, (w1,h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)),'constant')
    pad = (int(padh1), int(padw1))
    img2 = cv2.resize(img1, (128,128))
    return img1, img2, scale, pad


def _preprocess(img):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).permute((2, 0, 1))
    assert img.shape[1] == 3
    assert img.shape[2] == 192
    assert img.shape[3] == 192

    img = img.to(torch.device('cpu'))
    img = img.float() / 255.

    return img


def _decode_boxes( raw_boxes, anchors):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    boxes = torch.zeros_like(raw_boxes)

    x_center = raw_boxes[..., 0] / 192.0 * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / 192.0 * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / 192.0 * anchors[:, 2]
    h = raw_boxes[..., 3] / 192.0 * anchors[:, 3]

    boxes[..., 0] = x_center - w / 2.  # xmin
    boxes[..., 1] = y_center - h / 2.  # ymin
    boxes[..., 2] = x_center + w / 2.  # xmax
    boxes[..., 3] = y_center + h / 2.  # ymax

    num_keypoints = 7
    for k in range(num_keypoints):
        offset = 4 + k * 2
        keypoint_x = raw_boxes[..., offset] / 192.0 * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / 192.0 * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


def _tensors_to_detections(raw_box_tensor, raw_score_tensor, anchors):
    """The output of the neural network is a tensor of shape (b, 2016, 18)
    containing the bounding box regressor predictions, as well as a tensor
    of shape (b, 2016, 1) with the classification confidences.
    """
    assert raw_box_tensor.ndimension() == 3
    assert raw_box_tensor.shape[1] == 2016
    assert raw_box_tensor.shape[2] == 18

    assert raw_score_tensor.ndimension() == 3
    assert raw_score_tensor.shape[1] == 2016
    assert raw_score_tensor.shape[2] == 1

    assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

    detection_boxes = _decode_boxes(raw_box_tensor, anchors)

    thresh = 100
    raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
    detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = detection_scores >= 0.75

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
        output_detections.append(torch.cat((boxes, scores), dim=-1))

    return output_detections


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)


def _weighted_non_max_suppression(detections):
    """
    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.
    """
    if len(detections) == 0: return []

    output_detections = []
    num_coords = 18
    min_suppression_threshold = 0.5
    # Sort the detections from highest to lowest score.
    remaining = torch.argsort(detections[:, num_coords], descending=True)

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.clone()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :num_coords]
            scores = detections[overlapping, num_coords:num_coords + 1]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(dim=0) / total_score
            weighted_detection[:num_coords] = weighted
            weighted_detection[num_coords] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections


def denormalize_detections(detections, scale, pad):
    """ maps detection coordinates from [0,1] to image coordinates

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.
    """
    detections[:, 0] = (detections[:, 0] * 192 - pad[1]) * scale
    detections[:, 1] = (detections[:, 1] * 192 - pad[0]) * scale
    detections[:, 2] = (detections[:, 2]  * 192 - pad[1]) * scale
    detections[:, 3] = (detections[:, 3]  * 192 - pad[0]) * scale

    detections[:, 4::2] = (detections[:, 4::2] * 192 - pad[1]) * scale
    detections[:, 5::2] = (detections[:, 5::2]  * 192 - pad[0]) * scale

    return detections


def draw_detections(img, detections, with_keypoints=True):
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

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
    # cv2.imshow("points", img)
    return img

def detect():
    # gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    palm_detector = BlazePalm_QAT().to(torch.device('cpu'))
    palm_detector.load_state_dict(torch.load(r"E:\pytorch_QAT\BlazePalm_new_join\weights\  BlazePalm_QAT_8.pth", map_location=torch.device('cpu')))
    anchors = generate_anchors(SSD_ANCHOR_OPTIONS)
    anchors = torch.tensor(anchors, requires_grad=False)
    palm_detector.min_score_thresh = .75
    dataset_root = r'E:\datasets\hand_datasets\handpose_x\handpose_datasets_v1-2021-01-31\val_hand_v2'
    imgs_path = glob.glob(os.path.join(dataset_root, "*.jpg"))
    mirror_img = False
    with torch.no_grad():

        for i in range(len(imgs_path)):
            frame = cv2.imread(imgs_path[i])
            if mirror_img:
                frame = np.ascontiguousarray(frame[:,::-1,::-1])
            else:
                frame = np.ascontiguousarray(frame[:,:,::-1])

            img, img2, scale, pad = resize_pad(frame)
            # print("pad:", pad)
            # cv2.imshow("ssss", img1)
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).permute((2, 0, 1))
            img = img.unsqueeze(0)
            assert img.shape[1] == 3
            assert img.shape[2] == 192
            assert img.shape[3] == 192
            img = img.to(torch.device('cpu'))
            img = img.float() / 255.
            with torch.no_grad():
                out = palm_detector(img)
            detections = _tensors_to_detections(out[1], out[0], anchors)
            normalized_palm_detections = []
            for i in range(len(detections)):
                faces = _weighted_non_max_suppression(detections[i])
                faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 18 + 1))
                normalized_palm_detections.append(faces)

            palm_detections = denormalize_detections(normalized_palm_detections[0], scale, pad)
            draw_detections(frame, palm_detections)
            cv2.imshow("WINDOW", frame[:, :, ::-1])

            if cv2.waitKey(1000) == ord('q'):
                break
        # capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()