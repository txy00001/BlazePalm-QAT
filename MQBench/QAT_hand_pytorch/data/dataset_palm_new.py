import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import glob
import json
from tqdm import tqdm
from data.data_augment_new import resize_pad, resize_landmarks, draw_landmarks

def detection_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


class PalmDatasetv1(data.Dataset):
    def __init__(self, dataset_root):

        self.imgs_path = []
        self.words = []
        print("dataset_root:",dataset_root)
        self.imgs_path = glob.glob(os.path.join(dataset_root,"*.jpg"))
        self.landmark_idxs = ['0','1','2','5','9','13','17']
        # 处理标签文件
        for img_path in tqdm(self.imgs_path):
            # print("json_path:", img_path)
            json_path = img_path.replace(".jpg",".json")
            with open(json_path,'r') as fr:
                ann_obj = json.load(fr)
                infos = ann_obj['info']
                labels = []
                for info in infos:
                    bbox = info['bbox']
                    pts = info['pts']
                    label = [float(p) for p in bbox]
                    for landmark_idx in self.landmark_idxs:
                        # print("x:",pts[landmark_idx]["x"])
                        label.append(float(pts[landmark_idx]["x"]))
                        label.append(float(pts[landmark_idx]["y"]))
                    labels.append(label) #当前图片的每个box标注对象存储到labels中

            self.words.append(labels)


    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        bgr_img = cv2.imread(self.imgs_path[index])
        # imgc = bgr_img.copy()
        rgb_img = bgr_img[:,:,::-1] #bgr 2 rgb
        img, scale, pad = resize_pad(rgb_img,192,192)
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 19))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 19))
            # bbox
            annotation[0, 0] = label[0]  # x_min
            annotation[0, 1] = label[1]  # y_min
            annotation[0, 2] = label[2]  # x_max
            annotation[0, 3] = label[3]  # y_max

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[6]    # l1_x
            annotation[0, 7] = label[7]    # l1_y
            annotation[0, 8] = label[8]    # l2_x
            annotation[0, 9] = label[9]    # l2_y
            annotation[0, 10] = label[10]  # l3_x
            annotation[0, 11] = label[11]  # l3_y
            annotation[0, 12] = label[12]  # l4_x
            annotation[0, 13] = label[13]  # l4_y
            annotation[0, 14] = label[14]  # l5_y
            annotation[0, 15] = label[15]  # l5_y
            annotation[0, 16] = label[16]  # l6_y
            annotation[0, 17] = label[17]  # l6_y 4+14=18
            annotation[0, 18] = 1
            annotations = np.append(annotations, annotation, axis=0)

        annotations = np.array(annotations)
        landmarks = resize_landmarks(annotations, scale, pad)
        # draw_landmarks(img, landmarks, with_keypoints=True)
        anchor_x_center = (landmarks[:, 0] + landmarks[:, 2]) / 2
        anchor_y_center = (landmarks[:, 1] + landmarks[:, 3]) / 2
        anchor_w = abs(landmarks[:, 2] - landmarks[:, 0])
        anchor_h = abs(landmarks[:, 3] - landmarks[:, 1])
        landmarks[:, 0] = anchor_x_center
        landmarks[:, 1] = anchor_y_center
        landmarks[:, 2] = anchor_w
        landmarks[:, 3] = anchor_h
        landmarks[:, 0::2] /= width
        landmarks[:, 1::2] /= height
        landmarks[:, 18] = 1

        img = torch.from_numpy(img).permute((2, 0, 1))
        img = img.float() / 255.
        return img, landmarks


if __name__ == "__main__":
    trainset_loader = PalmDatasetv1(r'E:\datasets\hand_datasets\handpose_x\handpose_datasets_v1-2021-01-31\train_hand_v2')
    valset_loader = PalmDatasetv1(r'E:\datasets\hand_datasets\handpose_x\handpose_datasets_v1-2021-01-31\val_hand_v2')
    for i in range(trainset_loader.__len__(),valset_loader.__len__()):
        trainset_loader.__getitem__(i)
        valset_loader.__getitem__(i)

# draw_landmarks(img, landmarks[0], with_keypoints=True)
# cv2.imshow("hdetect", img)
# cv2.waitKey(1000)

