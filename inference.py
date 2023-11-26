import torch
import sys

from engine import train_one_epoch, evaluate
import utils as utils

import time
import glob

import bubble
import cv2
import os

def main():
    pathName = sys.argv[1]
    print("Doing " + pathName + "...")

    print('inference...')
    t = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('model.pt', map_location=device)
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
        cv2.imwrite(pathName + '/Results/' + os.path.basename(imgs[i]), img)

    dt = time.time() - t
    print(format(dt, ".2f"), ' sec for ', len(imgs), ' images')

if __name__ == "__main__":
    main()
