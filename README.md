# 3D Segmentation of Cleared Mouse Brains and Plaque.

#### Contributors: 
Veronika Valova:   <br>
Richard Harwood: https://github.com/RichardHarwood <br>

## Introduction
The bioimage analysis goal here was to segment the plaque from cleared mouse brains across different ages and investigate the influence of saffron on plaque density.
The biggest hurdle was segmenting the plaque the issue is that plaque only spanned a few pixels and there was noise that looked similar to plaque (a few white pixels clumped together). That being said it was not difficult to identify the plaques with the human eye. 

We tried a "recipe" of thresholding, remove objects, watershed but this did not perform great. We then tried cellpose - this was promising. We then decided that any generalist machine learning workflow would beneift from trained data so we segmented 5 slices to start with. We didnt see massive gains with cellpose but when we tried with Stardist we got really good segmentation. 

Here, we provide a drop down which documents the Python code. Upon publication all images will be publicly avalaible:
We note that we had to do very little beyond the example on the Stardist github, so we thank the authors of Stardist immensily. 

#### Python workflow to train star dist and segment unseen data (note that Juypter notebooks are also stored in this project)

**Import packages** 
```python 
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D
from stardist.models import StarDist3D
import tifffile 
from skimage import img_as_uint
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
from tqdm import tqdm
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

np.random.seed(6)
lbl_cmap = random_label_cmap()
```
**Define directories that contain images and masks and run some checks and normalistaions
*Note that the images are as a stack and the raw inage is 8 bit gray scale and the masks each plaque is a unique object)*

```python
X = sorted(glob('E:/StarDist/images/*.tif'))
Y = sorted(glob('E:/StarDist/masks/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

X = list(map(tifffile.imread,X))
Y = list(map(tifffile.imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]
```
**Split into training and testing data**
