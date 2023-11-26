# BubbleFinder
Easy and robust deep learning bubble segmentation

This code is based on the torchvision tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

main.py runs the training on your own trainingset
the dataloader expects four folders:
BoundingBoxes: text files with a list of the bounding boxes of all bubbles [upper left x coordinate, upper left y coordinate, width, height]
Images: images (.png format, but of course you can change this)
Labels: text files with a list of labels per object (Bubble in all cases probably :))
Masks: text files with a list of linear indices of the mask of each bubble

the model is saved in the home folder (make sure not to overwrite it!)

inference.py runs the model on any image that you want
you can set a threshold for the detection

track.py performs both inference and tracking and colors the bubbles according to their ID

The rest are helperfiles

A manuscript using this code has been submitted to Industrial & Engineering Chemistry Research 


- trackpy has a bug that gives the following error: TypeError: mean() got an unexpected keyword argument 'level'
  here is a fix, if they haven't already fixed it themselves: https://github.com/soft-matter/trackpy/pull/740/commits/a3d4322876110a22807cc3440f422d8f85c1590a



