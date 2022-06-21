# Sorghum - 100 Cultivar Identification - FGVC 9

## 1. The Overview
The objective of this notebook is to setup a Convolutional Neural Nets (CNN) Model, train it, and submit it to the Kaggle Sorghum - 100 Cultivar Identification Competition.

You can find the complete overview of the competition and the datasets by clicking <a href='https://www.kaggle.com/competitions/sorghum-id-fgvc-9'>HERE</a>.

<b>Lets make a short recap of the Sorghum - 100 Cultivar Identification - FGVC 9 Competition.</b>

The Sorghum-100 dataset is a curated subset of the RGB imagery captured during the TERRA-REF experiments, labeled by cultivar. This data could be used to develop and assess a variety of plant phenotyping models which seek to answer questions relating to the presence or absence of desirable traits (e.g., "does this plant exhibit signs of water stress?''). In this contest, we focus on the question: "What cultivar is shown in this image?''

The objective of the competition is to set a Machine Learning Model which will predicts the correct sorghum cultivar given a sorghum picture.

## 2. The Dataset
<img src="https://i.imgur.com/dlOnvRn.png">

The Sorghum-100 dataset consists of 48,106 images and 100 different sorghum cultivars grown in June of 2017 (the images come from the middle of the growing season when the plants were quite large but not yet lodging -- or falling over). In the above image, we show a sample of images from four different cultivars. Each row includes six images from different dates in June. This figure highlights the high inter-class visual similarity between the different classes, as well as the high variability in the imaging conditions from one day to the next, or even over the course of a day.

Each image is taken using an RGB spectral camera taken from a vertical view of the sorghum plants in the TERRA-REF field in Arizona.

You can download the entire dataset by clicking <a href='https://www.kaggle.com/competitions/sorghum-id-fgvc-9/data'>HERE</a>.

## 3. Convolutional Neural Nets (CNN)
<img src="https://miro.medium.com/max/1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg">
A Convolutional Neural Network (CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other,  making it the best candidate model for our competition.

The architecture of a CNN is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.

## 4. Transfer Learning
The basic premise of transfer learning is simple: take a model trained on a large dataset and transfer its knowledge to a smaller dataset. We can train it from scratch, utilizing only its architecture, or freeze some layers and fine tune it.
The are plenty of pre-trained models out there, we are going to use the EfficientNetB2. 

## 5. The Plan of Attack
<img src='https://user-images.githubusercontent.com/32513366/71764203-797da800-2ec3-11ea-9eb9-8bdca4f45152.jpg' width=400 height=400>
We are going to build our model with TensorFlow - Keras, which makes the whole proccess of ensambling and training Neural Nets way easier.

The whole process will consist in 5 parts:
* Load Data
* Data Augmentation
* Compile the Model
* Monitor the Training
* Predict the Outcomes

A very important step in training a good CNN model is the Data Augmentation, which will prevent overfitting. Besides the built-in data augmentation tools, we are gonna use the Cutmix tool to enhance our agumentation.

## 6. The Model
<b>You can access the full notebook <a href='https://github.com/nicholascomuni/Sorghum-100-Cultivar-Identification-EfficientNet/blob/master/KaggleSorghum100.ipynb'>HERE</a>.
<img src='img/model.png](https://raw.githubusercontent.com/nicholascomuni/Sorghum-100-Cultivar-Identification-EfficientNet/master/Img/model.png)' width=400></b>


Epoch 1/10
1480/1480 - 5461s - loss: 4.0380 - accuracy: 0.1098

Epoch 2/10
1480/1480 - 5140s - loss: 2.7756 - accuracy: 0.4004

Epoch 3/10
1480/1480 - 5140s - loss: 1.9511 - accuracy: 0.6333

Epoch 4/10
1480/1480 - 5119s - loss: 1.4336 - accuracy: 0.7557

Epoch 5/10
1480/1480 - 5136s - loss: 1.2705 - accuracy: 0.7837

Epoch 6/10
1480/1480 - 5135s - loss: 1.3065 - accuracy: 0.7738

Epoch 7/10
1480/1480 - 5157s - loss: 1.2264 - accuracy: 0.7862

Epoch 8/10
1480/1480 - 5132s - loss: 1.0509 - accuracy: 0.8149
...

## 7. The Results
Submitting the outcomes to Kaggle we got 0.743 (74.3%) in the Public Score, which is great!
