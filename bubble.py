import numpy as np
import torch
from PIL import Image
import glob

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import transforms as T
from torchvision import transforms as TT

import random
import cv2


class BubbleDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(glob.glob(root + "/Images/*.png")))
        self.masks = list(sorted(glob.glob(root + "/Masks/*.txt")))
        self.bbs = list(sorted(glob.glob(root + "/BoundingBoxes/*.txt")))
        self.labels = list(sorted(glob.glob(root + "/Labels/*.txt")))


    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        bbs_path = self.bbs[idx]
        label_path = self.labels[idx]

        #print(img_path)
        img = Image.open(img_path).convert("RGB")
        img_size = (img.width, img.height)

        labels: [str] = []
        with open(label_path) as f:
            for i, line in enumerate(f):
                labels.append(line.strip())
        num_objs = len(labels)

        boxes: [[int]] = []
        with open(bbs_path) as f:
            for i, line in enumerate(f):
                spline = line.split('\t')
                spline.pop()
                for s in spline:
                    if s == '':
                        print(bla)
                spline = [int(s) for s in spline]
                boxes.append(spline)

        masks = np.zeros((num_objs,) + (img.height, img.width))
        with open(mask_path) as f:
            for i, line in enumerate(f):
                spline = line.split('\t')
                spline.pop()
                spline = [int(s) for s in spline]
                mask = np.zeros(img_size)
                rmask = np.ravel(mask)
                rmask[spline] = 1
                mask = np.reshape(rmask, img_size)
                mask = np.swapaxes(mask, 0, 1)

                masks[i, :, :] = mask

                # Test if masks and images align
                #rgb_mask = random_color_masks(mask)
                #cv2.imwrite('test/Mask' + f"{i:06d}.jpg", rgb_mask)
                #cv2.imshow('', rgb_mask)
                #cv2.waitKey(0)

        # Test all masks
        # np_img = np.array(img)
        # for i in range(num_objs):
        #     rgb_mask = random_color_masks(np.array(masks[i, :, :]))
        #     np_img = cv2.addWeighted(np_img, 1, rgb_mask, 0.5, 0)
        #cv2.imwrite('test/' + f"{idx:06d}.jpg", np_img)


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=500)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


BUBBLE_CATEGORY_NAMES = ['__background__', 'bubble']


def get_prediction(img_path, model, device, threshold=0.7):
    img = Image.open(img_path)
    transform = TT.ToTensor()
    img = transform(img)
    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [BUBBLE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in
                  list(pred[0]['boxes'].cpu().detach().numpy())]
    pred_scores = pred_score[:pred_t + 1]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class, pred_scores


def random_color_masks(image, single_color=False):
    colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
              [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    if single_color:
        r[image == 1], g[image == 1], b[image == 1] = colors[8]
    else:
        r[image == 1], g[image == 1], b[image == 1] = colors[random.randrange(0, 10)]

    colored_mask: np.stack = np.stack([r, g, b], axis=2)
    return colored_mask


def instance_segmentation(img_path, model, device, threshold=0.7, rect_th=1, text_size=1, text_th=2):
    masks, boxes, pred_cls, pred_scores = get_prediction(img_path, model, device, threshold=threshold)
    img = cv2.imread(img_path)
    for i in range(len(masks)):
        rgb_mask = random_color_masks(masks[i], single_color=False)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        gray = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 0, 0), thickness=rect_th)
    return img, pred_cls, pred_scores, masks
