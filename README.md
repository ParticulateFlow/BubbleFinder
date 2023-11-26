# BubbleFinder
Easy and robust deep learning bubble segmentation

This code is based on the torchvision tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

main.py runs the training on your own trainingset
the dataloader expects four folders:
BoundingBoxes: text files with a list of the bounding boxes of all bubbles [upper left x coordinate, upper left y coordinate, width, height]
Images: images (.png format, but of course you can change this)
Labels: text files with a list of labels per object (Bubble in all cases probably :))
Masks: text files with a list of linear indices of the mask of each bubble

the model is saved in the home folder (make sure not to overwrite it)

inference.py runs the model on any image that you want
you can set a threshold for the detection

The rest are helperfiles



