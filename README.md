# ESRGAN_Proba-V
Tackling Proba-V competition with a modified version of ESRGAN

## About Proba-V
Proba-V is a miniaturized ESA satellite tasked with a full-scale mission: to map land cover and vegetation growth across the entire earth every two days. It is the latest in ESA’s Proba series of minisatellites, among the smallest satellites launched by the agency.<br/>
It operates on a sun-synchronous near polar Earth orbit at about 820Km guaranteeing the required swath of 2250Km with an instrument field of view of 102 degrees, compatible with the geographical coverage.<br/>
The Proba-V satellite carries a new Vegetation instrument, as single operational payload.<br/>
In the frame of the In-Orbit Demonstration, the Proba-V platform also flies 5 technological payloads:
* X-Band transmitter based on GaN RF amplifier
* Energetic Particle Telescope EPT
* Automatic Dependent Surveillance Broadcast (ADS-B) receiver
* SATRAM radiation monitoring system, complementing EPT
* HERMOD (fiber optic connectivity in-situ testing)

The Proba-V mission provides multispectral images to study the evolution of the vegetation cover on a daily and global basis.<br/>
The ‘V’ stands for Vegetation, and the mission is extending the data set of the long-established Vegetation instrument, flown as a secondary payload aboard France’s SPOT-4 and SPOT-5 satellites launched in 1998 and 2002 respectively.<br/>
The Proba-V mission has been developed in the frame of the ESA General Support Technology Program GSTP and the contributors to the Proba-V mission are Belgium and Luxembourg.<br/>
For more details: https://www.esa.int/Applications/Observing_the_Earth/Proba-V

## Proba-V Super Resolution Competition
Proba-V Super Resolution Competition has been launched on the 1st of November 2018 with a timeline of 8 months to get an end at the 1st of June 2019. <br/>
In this competition multiple images are given of each of the 74 Earth locations and challengers were asked to develop an algorithm to fuse them together into a single image. <br/>
The result will be a “Super-Resolved” image that is checked against a “High Resolution” image taken from the same satellite PROBA-V. <br/>
The main goal of this competition is to construct such high-resolution images by fusion of the more frequent 300m images.<br/>
This process, which is known as Multi-image Super Resolution has already been applied to satellite before. <br/>
The images provided for this challenge are not artificially degraded, but are real images recorded from the very same scene, just at different resolutions and different times. Any improvements on this data-set might be transferable to larger collections of remote sensing data without the need to deploy more expensive sensors or satellites, as resolution enhancement can happen post-acquisition.<br/>
So, the goal is the enhancement of the vision PROBA-V and helping researchers advance the accuracy on monitoring earths vegetation growth.<br/>

## DATASET
The Data is composed of radiometrically and geometrically corrected TOA (Top of Atmosphere) reflectance’s for the RED and NIR spectral bands at *300m* and *100m* resolution.<br/>
The *300m* resolution data delivered as *128x128* grey-scale pixel images, the *100m* resolution data as *384x384* grey-scale pixel images.<br/>
The bit-depth of the images is 14, but the are saved in 16-bit .png format.<br/>
The dataset contains 1450 scenes, which are split into 1160 scenes for training and 290 for testing.<br/>
Each scene contains at least 9 low resolution images and max of 34 images (LR*), their respective status map (QM*) and one high resolution (HR*) and its respective status map (SM*).<br/>
Once you unzip the probav_data.zip you will get the following path: 
* Proba-V / test / NIR / scenes / LR + QM
* Proba-V / test / RED / scenes / LR + QM
* Proba-V / train / NIR / scenes / LR + QM + HR + SM
* Proba-V / train / RED / scenes / LR + QM + HR + SM
* Proba-V / norm.csv

## My own Approach
I will detail my reasoning and finding when attempting the Proba-V super resolution competition. (competition is already over).
This project was made in 2 steps:
1. Preparing the data: Images preprocessing
2. Training the model and then deploying it (Test)
After looking on previous works that been done on Proba-V super resolution competition and while attempting to make a new approach I decide to use a modified version of ESRGAN (Enhanced Super Resolution Gan) which is already a modified version of SRGAN in order to tackle the problem.<br/>
Details can be found in these two links: 
* [SRGAN] (https://arxiv.org/pdf/1609.04802.pdf)
* [ESRGAN] (https://arxiv.org/pdf/1809.00219.pdf)
<p align="center">
  <img src="figures/Capture1.PNG">
</p>
While creating ESRGAN model, I skipped the 2nd aspect: which is the use of RAGAN.<br/>
Next version of the project will contain RAGAN.

## Preparing the Data: (Preprocessing.ipynb)
In order to have one candidate image using multiple low resolution (LR) images, some preprocessing steps were necessary.<br/>
<p align="center">
  <img src="figures/aggregation.png">
</p>

Some images in the dataset are partially obstructed (Clouds, Ice…) and up to *25%* of pixels can be concealed (*40%* for low resolution images) thus, increasing the need of using multiple low-resolution images.<br/>
The images are a representation of two spectral bands (RED and NIR). <br/>
The images are stored in a uint16 bit format as PNGs and are supposed to be of 14bit-depth.<br/>
So, my initial pipeline process was the following:
#### For Low Resolution (Train & Test):
1. Loading the images as uint16
2. Using the norm.csv to normalize cPSNR
3. Parsing the dataset and checking for high pixels in order to remove them, since the images are stored in a uint16.
4. Decomposing the status map (QM) into 8 patches and searching for the best score among all QMs to generate candidate patches that will be used to construct the candidate image (Respective LR patches to QM patches).
5. Saving the created candidate LR as float64 image and shifting it to 14bits.
6. Creating an RGB representation of a grey-level image using the gray2rgb function.
7. Saving LRs as np.array
**PS**: Since I’m low on memory I processed each band on it own and then I combine them into a single np.array ‘LR_train.npy’ and ‘LR_test.npy’
#### For High Resolution (Only Train):
