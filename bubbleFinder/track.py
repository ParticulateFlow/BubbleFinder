from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt

# Optionally, tweak styles.
mpl.rc('figure', figsize=(10, 5))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
import trackpy as tp
import time
import sys
import glob
import cv2
import os
import pickle

import torch

import bubble


def main():

    pathName = sys.argv[1]
    print("Doing " + pathName + "...")

    imgs = list(sorted(glob.glob(pathName + "*.png")))

    inference = True

    if inference:
        print('inference...')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load('model.pt', map_location=device)
        model.to(device)
        model.eval()

        columns = ['y', 'x', 'mass', 'frame', 'bubble']
        all_bubbles = pd.DataFrame(columns=columns)
        for i in range(0, len(imgs)):
            print('Image ' + str(i))
            img, pred_classes, pred_scores, masks = bubble.instance_segmentation(imgs[i], model, device, threshold=0.9)
            dt = time.time()
            masks = masks.astype(int)

            bubbles: [[float]] = []
            index: [int] = []
            for b, mask in enumerate(masks):
                index.append(b)
                area = np.count_nonzero(mask)
                x_center, y_center = np.argwhere(mask == 1).sum(0) / area
                bubbles.append([x_center, y_center, area, i, b])

            bubblespd = pd.DataFrame(bubbles, columns=columns, index=index)
            all_bubbles = pd.concat([all_bubbles, bubblespd])

            if not os.path.exists(pathName + '/Masks/'):
                os.makedirs(pathName + '/Masks')
            np.save(pathName + '/Masks/Masks' + f"{i:06d}.npy", masks)
            print('time ' + format(time.time() - dt, '.2f') + ' seconds')


        all_bubbles.to_pickle(pathName + "all_bubbles.pkl")
    else:
        all_bubbles = pd.read_pickle(pathName + "all_bubbles.pkl")

    print(all_bubbles)

    t = tp.link(all_bubbles, 30, memory=30)
    em = tp.emsd(t, 1, 1)
    print(em.get(1))

    t1 = tp.filter_stubs(t, 30)
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())
    t.to_pickle(pathName + "tracked_bubbles.pkl")
    print(t)

    plt.figure()
    ax = tp.plot_traj(t)
    ax.set_aspect('equal', adjustable='box')
    plt.draw()
    plt.show()

    for i in range(0, len(imgs)):
        print('Image ' + str(i))
        img = cv2.imread(imgs[i])
        masks = np.load(pathName + '/Masks/Masks' + f"{i:06d}.npy")
        this_frame = t.loc[t['frame'] == i]

        for b, row in this_frame.iterrows():
            rgb_mask = bubble.random_color_masks(masks[row['bubble']], int(row['particle']) % 10)
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)

        if not os.path.exists(pathName + '/Results/'):
            os.makedirs(pathName + '/Results/')
        cv2.imwrite(pathName + '/Results/Image' + f"{i:06d}.png", img)
        #cv2.imshow('', img)
        #cv2.waitKey(0)

if __name__ == "__main__":
    main()
