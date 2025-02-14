import cv2
import torch
import numpy as np
import os
import time
import glob
from functions import bubble

pathName = "./data/images/"
pathModel = "./data/models/model_125images_80bubbles_80minDistance_40aperture_30epochs.pt"

print('test...')
t = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(pathModel, weights_only=False, map_location=device)
model.to(device)
model
model.eval()

imgs = list(sorted(glob.glob(pathName + '*.tiff')))

for i in range(0, len(imgs)):
    try:
        print('Image ' + str(i) + ' : ' + os.path.basename(imgs[i]))
        img, pred_classes, pred_scores, masks = bubble.instance_segmentation(imgs[i], model, device, threshold=0.9)
        print(len(masks), ' bubbles found')

        if not os.path.exists(pathName + '/Results_2/'):
            os.makedirs(pathName + '/Results_2/')
        cv2.imwrite(pathName + '/Results_2/' + os.path.basename(imgs[i]), img)
    except:
        pass
    
dt = time.time() - t
print(format(dt, ".2f"), ' sec for ', len(imgs), ' images')
