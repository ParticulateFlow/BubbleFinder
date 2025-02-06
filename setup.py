from setuptools import setup

setup(
   name='BubbleFinder',
   version='1.0',
   description='Easy and robust deep learning bubble segmentation',
   author='t.a.m.homan@tue.nl',
   author_email='t.a.m.homan@tue.nl',
   url='https://pubs.acs.org/doi/10.1021/acs.iecr.3c04059',
   install_requires=[
        'matplotlib', 
        'numpy', 
        'pandas', 
        'Pillow'
        'pycocotools',
        'scipy',
        'torch',
        'torchvision',
        'tqdm',
        'trackpy',
        'opencv-python']
)