# ATLAS GSoC

Compression of jet events data using Autoencoders.

The data consisted of 4 features namely `m`, `pt`, `phi`,  `eta` which were compressed to 3 using autoencoders architecture.

## Project Structure

* **notebooks** folder contain Jupyter Notebooks which include all the experiments done.
* **plots** contain png images of all plots
* **src** contains Python scripts for easy use
* **trained_model** has the trained model (pytorch state dict).

## Problem Approach

**Exploratory Data Analysis:** Started out with getting information about data and came up with the following observations:

1. The range of data was very different, 2 features ranged between -3 to 3 and -4 to 4. Other 2 features ranged from 0 to 120,000 and 20,000 to 700,00.
2. There was also a lot of skewness in two features.
3. The distribution of train and test set were similar.

**Data Pre-processing:** To get good performance, the data had to be brought down to relatively similar scale. I started out with the Standard Scalar function (0 mean, 1 standard deviation) and also tried various custom scaling functions as well.
Custom scaling function gave better performance.

**Model Architecture:** I decided to use the same architecture as used in the [project](https://github.com/Skelpdar/HEPAutoencoders) and make improvements upon that. Used `LeakyReLU` activation function instead of `Tanh`

**Model Training:** The following settings were used during training:

* **Batch Size:** 1024
* **Optimizer:** ADAM with Weight Decay with varying learning rates
* **Loss function:** Mean Squared Error
