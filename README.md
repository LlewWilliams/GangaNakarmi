# "A crowdsourced approach to documenting users’ preferences for landscape attributes in the proposed Appalachian Geopark Region in West Virginia" by Ganga Nakarmi
The Open Access manuscript will be available here - > 

The following document contains a workflow to reproduce the results in the above paper. If you find these codes useful in your research, please cite!

## Credits
Ganga Nakarmi, School of Design and Community Development, West Virginia University, WV
gn0001@mix.wvu.edu

and

Llew Williams, Optimist Consulting, WV, providing coding assistance
llew0williams@gmail.com

## Running the code
This is a python workflow that requires the installation of some third-party libraries, some of them are older versions. We recommend that you use a virtual environment and the conda platform for installing and managing dependencies. We used Google colab to increase performance. We used Visual Studio code for editing, though that is not a requirement. Reproducing the results in the paper will require some experience with coding and running python scripts.  

## Data Set
The dataset consisted of 2,945 jpgs that were taken between 2010 and 2020 in the proposed Geopark region

## Workflow to Download Photos
### Obtaining Crowdsourced image information from Flickr for a specic Geographic region
Usage: 

We prepared a python file to connect to Flickr and download metadata based on our search criteria. This file includes an algorithm to break up the area of interest into smaller bounding boxes to limit recordset size to stay within Flickr data limits. 
```
flickr_crowdsource_data_download.py
```

Example Usage:

We ran this code in Visual Studio after setting up path info and credentials.

### Downloading Images from Flickr After Vetting
Usage: 

After trimming the data set manually we used the remaining data in a csv file with this script to download the actual .jpg images from Flickr. 
```
download_jpg_from_flickr.py
```

Example Usage:

We ran this code in Visual Studio after setting up path info and credentials.

## Preparing a python environment with required libraries 
### Create a virtual environment using Anaconda

```
conda create --name tfpy35 python=3.5
conda activate tfpy35
```
### Install the pydensecrf package from conda forge

```
conda config --add channels conda-forge
conda install pydensecrf
conda config --remove channels conda-forge
```
### Install python libraries

```
pip install Cython
pip install numpy scipy matplotlib scikit-image scikit-learn
pip install joblib
pip install tensorflow tensorflow_hub
conda install opencv
```
## Workflow to Process Images With A Deep Neural Network
### Create ground truth images for training and validating a model
Usage: 

```
create_groundtruth.py
```

Example Usage:

We ran this code in Visual Studio after setting up path info.

Outputs: 
1. A .png file consisting of 3 images, the original image, manual labeling, CRF pixel-level predictions
2. A .mat (matlab format) file with the fields: 'sparse' (manual labeling); 'labels' (the labels used in classification); and 'class' (CRF pixel-level predictions)

Example python code for loading and printing the .mat files:
```
from scipy.io import loadmat
dat = loadmat('ourMatFile.mat')
print(dat.keys())
```

### Retraining a DCNN network using our ground truth images for use in image recognition
Usage: 

We performed the retraining on Google Colab to take advantage of a GPU and more processing power
```
retrain.py
```

Example usage:
This is an example of running on Colab in a Colab Notebook
```
from google.colab import drive
drive.mount('/content/gdrive')
#trying this here to stop using compat library
%tensorflow_version 1.x
!pip install --upgrade tensorflow-hub

%cd "/content/gdrive/My Drive/imageDirectory"
!python retrain.py --image_dir='autoClassifiedNewModelName'
```
### Testing Image Recognition with our DCNN model
Usage: 

We prepared a set of image tiles using create_groundtruth.py then tested the performance of our model. The script compiles statistics about average performance. I will also print out a confusion matrix, like those included in the published paper. Mean accuracy, Mean Probability, and Mean FScore are printed to the screen. 
```
image_recog.py
```

Example usage:
This is an example of running on Colab in a Colab Notebook
```
from google.colab import drive
drive.mount('/content/gdrive')

#%tensorflow_version 1.x < - this used to work. Google updated colab - work around below is working today 2022-09-26
!pip install tensorflow==1.15.2
!pip install --upgrade tensorflow-hub

%cd "/content/gdrive/My Drive/SharedWithGanga/WorkingDirectory"

!pip3 install scipy==1.2.1

!pip install git+https://github.com/lucasb-eyer/pydensecrf.git

!python test_imrecog.py 
```

### Using our DCNN model to perform image classifiction of a set of images
Usage: 

This script walks though a directory and performs segmantic segmentation on all images in the folder. Pixel-level metrics is output to a csv file for each label in each image. This data is easily uploaded to a spreadsheet or database for aggregate analysis. The script also outputs a .png file for each image containing the original image, the CNN, and the CRF results. 
```
semseg_cnn_crf.py
```

Example usage:
This is an example of running on Colab in a Colab Notebook
```
from google.colab import drive
drive.mount('/content/gdrive')

#%tensorflow_version 1.x < - this used to work. Google updated colab - work around below is working today 2022-09-26
!pip install tensorflow==1.15.2
!pip install --upgrade tensorflow-hub

%cd "/content/gdrive/My Drive/SharedWithGanga/WorkingDirectory"

!pip3 install scipy==1.2.1

!pip install git+https://github.com/lucasb-eyer/pydensecrf.git

!python semseg_cnn_crf.py 
```


