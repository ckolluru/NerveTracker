# NerveTracker
Software to track fibers in block-face microscopy images.

The software provides the user with the following options
1. Load a 3D-MUSE image dataset (as a set of 2D PNG files)
2. Visualize the stack by moving across slices
3. Run one of two available flow estimation methods (optic flow and structure tensor analysis)
4. Cluster generated tractograms containing several streamlines
5. Interactive editing of tractograms
6. Save tractograms to disk for loading in the future.

Tractogram based metrics
Tool for comparing two tractograms based on a closest neighbor distance metric is available from the GUI.
Tool for computing Dice overlap between the tractogram and a binary mask is available from the script validate_tractograms.py

## Installation

1. Clone this repository
```git clone https://github.com/ckolluru/NerveTracker.git```

2. Navigate to the repository folder

3. Create a conda environment
```conda create --name tracking --file requirements.txt```

4. Activate the environment
```conda activate tracking```

## Algorithms

There are two options to run the tractography analysis, optic flow and structure tensor analysis.
Parameters to both algorithms can be set by the user and default values are provided. 

The optic flow analysis uses the Lucas-Kanade algorithm, the specific implementation is from the opencv2 library. 
Please refer to the [library’s documentation](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html) for a description of the parameters. 
More information about the Lucas-Kanade algorithm are available in a set of [introductory videos online](https://www.youtube.com/watch?v=lnXFcmLB7sM&list=PL2zRqk16wsdoYzrWStffqBAoUY8XdvatV&index=1) (videos 1 to 6).

The software also provides the opportunity to analyze the dataset using the structure tensor approach. 
The images are generally down sampled by a suitable integer factor to create near-isotropic voxels in XYZ.
This also has the advantage of speeding up the calculations. An introduction to the structure tensor approach is available [here](http://people.compute.dtu.dk/vand/notes/ST_intro.pdf). 

Seeds per pixel indicates the number of seed points that will be created. The default is one seed point every 100 pixels, or 0.01 seeds per pixel. 
Clustering threshold is a metric that is used by the [Quickbundles algorithm](https://doi.org/10.3389%2Ffnins.2012.00175) to cluster streamlines to generate a compact representation and visualization of the tractogram. 


