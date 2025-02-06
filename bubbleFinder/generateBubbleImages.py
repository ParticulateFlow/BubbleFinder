import os
import cv2 as cv    #opencv-python
import glob
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

imagePathName = './SingleBubbleImages/Images/'
imageFileNames = list(sorted(glob.glob(imagePathName + "*.tiff")))
maskPathName = './SingleBubbleImages/Masks/'
maskFileNames = list(sorted(glob.glob(maskPathName + "*.tiff")))

saveFolder = './generatedBubbleImages/'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

numSingleBubbles = len(imageFileNames)
imageSize = [1001, 601]
numBubbles = 60
numImages = 30

frame0 = cv.imread(imageFileNames[0])
subImageSize = frame0.shape
xmin = 1; xmax = imageSize[1] - subImageSize[1]
ymin = 1; ymax = imageSize[0] - subImageSize[0]
minDistance = 20
aperture = 95

savePath = saveFolder + str(numImages) + 'images_' + str(numBubbles) + 'bubbles_' + str(minDistance) + 'minDistance_' + str(aperture) + 'aperture/'
if not os.path.exists(savePath):
    os.makedirs(savePath)
    os.makedirs(savePath + '/BoundingBoxes/')
    os.makedirs(savePath + '/Masks/')
    os.makedirs(savePath + '/Labels/')
    os.makedirs(savePath + '/Images/')

for i in tqdm(range(numImages)):
    boundingBoxName = os.path.join(savePath + '/BoundingBoxes/' + 'BoundingBox%06d.txt' % i)
    fileIDBoundingBox = open(boundingBoxName, "w")
    maskName = os.path.join(savePath + '/Masks/' + 'Mask%06d.txt' % i)
    fileIDMask = open(maskName, 'w')
    labelName = os.path.join(savePath + '/Labels/' + 'Label%06d.txt' % i)
    fileIDLabel = open(labelName, 'w')

    Image = np.ones(imageSize) * 170/255
    BBs = []
    Labels = []
    Masks = []
    LabelMask = np.zeros(imageSize)
    areaSum = 0

    perm = np.random.permutation(numSingleBubbles)[:numBubbles]

    counter = 0
    xs = []
    ys = []
    while counter < numBubbles:
        xc = np.random.randint(xmin + aperture, high=xmax - aperture)
        yc = np.random.randint(ymin + aperture, high=ymax - aperture)

        if counter > 0:
            d = np.linalg.norm(np.array([xc, yc]) - np.column_stack((xs, ys)), axis=0)
            if np.min(d) <= minDistance:
                continue
            
        xs.append(xc)
        ys.append(yc)

        im = cv.imread(imageFileNames[perm[counter]], cv.IMREAD_GRAYSCALE)
        im = im.astype(float) / 255.0

        msk = cv.imread(maskFileNames[perm[counter]], cv.IMREAD_GRAYSCALE)
        msk = msk.astype(float) / 255.0
        msk = msk > 0.5

        im = np.multiply(msk, im)

        mask = np.zeros(imageSize)
        mask[yc : yc + subImageSize[1], xc: xc + subImageSize[0]] = msk

        vertical = np.any(mask, axis=1)
        horizontal = np.any(mask, axis=0)
        row1 = np.argmax(vertical)
        row2 = len(vertical) - np.argmax(vertical[::-1]) - 1
        column1 = np.argmax(horizontal)
        column2 = len(horizontal) - np.argmax(horizontal[::-1]) - 1

        img = np.zeros(imageSize)
        img[yc : yc + subImageSize[1], xc: xc + subImageSize[0]] = im
        k = np.where(img != 0)

        Image[k] = img[k]

        fileIDBoundingBox.write('%d\t%d\t%d\t%d\n' % (column1, row1, column2, row2))
        linear_indices = np.ravel_multi_index(np.where(mask), mask.shape)
        fileIDMask.write('\t'.join(map(str, linear_indices)) + '\n')
        fileIDLabel.write('Bubble\n')
        counter +=1

    imageName = os.path.join(savePath, 'Images', 'Image{:06d}.png'.format(i))
    cv.imwrite(imageName, Image*255)
    # cv.imshow('', Image)
    # cv.waitKey(0)
