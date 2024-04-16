# BubbleFinder
Easy and robust deep learning bubble segmentation

This repository contains the files used to generate the data in the paper: Deep learning bubble segmentation on a shoestring (https://doi.org/10.1021/acs.iecr.3c04059)

I've updated the code with data from the Million bubble dataset from Yucheng Fu (https://data.mendeley.com/datasets/mxnzxzc6v7/1, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)), such that you can test the code without your own data or rewriting the dataloader. I've taken the first 10000 bubbles out of the .mat file for convenience and created a mask for each image with corresponding names in the folder masks.

This neural network is based on the torchvision tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

code
----------
singleBubbleImages.zip - contains the first 10000 single bubble images from the Million bubble dataset with corresponding masks

generateBubbleImages.py - generates multi-bubble images from the single bubble data. You can set the image size, the number of bubbles, the number of images generated, the minimum distance between the center of the bubbles and the aperture (border around the image without bubbles, used to increase the bubble density without chaning any of the other parameters)
The output is four folders containing
BoundingBoxes: text files with a list of the bounding boxes of all bubbles [upper left x coordinate, upper left y coordinate, width, height]
Images: images (.png format, but of course you can change this)
Labels: text files with a list of labels per object (Bubble in all cases probably :))
Masks: text files with a list of linear indices of the mask of each bubble

main.py - runs the training on your own trainingset. The model is saved in the home folder (make sure not to overwrite it!)

inference.py - runs the model on any image that you want. You can set a threshold for the detection.

track.py - performs both inference and tracking and colors the bubbles according to their ID. This file is used to create the supplemental video S2.

The rest are helperfiles

Know issues
----------
- trackpy has a bug that gives the following error: TypeError: mean() got an unexpected keyword argument 'level'
  here is a fix, if they haven't already fixed it themselves: https://github.com/soft-matter/trackpy/pull/740/commits/a3d4322876110a22807cc3440f422d8f85c1590a



