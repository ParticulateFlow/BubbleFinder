import torch
import sys

from functions.engine import train_one_epoch, evaluate
from functions import utils as utils

import time
import glob

from functions import bubble
import cv2 as cv
import os

def main():
    #pathName = sys.argv[1]
    #print("Doing " + pathName + "...")
    # pathName = './Inference/'
    pathName='//pfm-daten/scratch/K1Met-P3.1-Bubble-Flows/WP1-Single-Bubble/20240219-Video-2mm-90ml/20240219T1611_d2.0_90ml_250fps/variabel_size_images/125images_80bubbles_80minDistance_40aperture/Images/'
    pathModel = '//pfm-daten/scratch/K1Met-P3.1-Bubble-Flows/WP1-Single-Bubble/20240219-Video-2mm-90ml/20240219T1611_d2.0_90ml_250fps/variabel_size_images/125images_80bubbles_80minDistance_40aperture/model_10epochs.pt'

    print('inference...')
    t = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(pathModel, weights_only=False, map_location=device)
    model.to(device)
    model
    model.eval()

    imgs = list(sorted(glob.glob(pathName + '*.png')))

    for i in range(0, len(imgs)):
        print('Image ' + str(i) + ' : ' + os.path.basename(imgs[i]))
        img, pred_classes, pred_scores, masks = bubble.instance_segmentation(imgs[i], model, device, threshold=0.9)
        print(len(masks), ' bubbles found')
        print('average prediction scores - ', sum(pred_scores)/len(pred_scores))

        if not os.path.exists(pathName + '/Results/'):
            os.makedirs(pathName + '/Results/')
        cv.imwrite(pathName + '/Results/' + os.path.basename(imgs[i]), img)

    dt = time.time() - t
    print(format(dt, ".2f"), ' sec for ', len(imgs), ' images')

if __name__ == "__main__":
    main()
