# Speech Emotion Recognition Project for Intel Edge AI Scholarhip Challenge 2020 SpeechVINO study group


## Abstract:

Advances in technology have enabled machines and alogorithms to recognize various human emotions. Emotion recognition implies a huge social impact and has become more and more demanding in a variety of fields, from retail to healthcare. Except for the crucial part of emotion recognition in healthcare, which helps in the diagnosis of mental issues, by identification a pattern in emotional types, advertisement is another field, where emotion recognition is thriving. For example many businesses would like to know how customers respond to ads or products. Also, application of emotion recognition can be found in education, where applications can measure real-time learner responses to and engagement with educational content. This way the content of a lecture can be adapted appropriately and the application also serves as mean of measuring the effectiveness of the lecturer.

Therefore, AI has become an important tool for the above mentioned applications, offering great solutions not only to industry but also to social issues. Aiming to contribute to a continuoulsy growing field, which provides helpful applications not only in industry, but also in healthcare and society, our team opted for this speech emotion recognition project, with promising results. 


## Objective:

Considerig the importance of emotion recognition in modern societies and industry, the objective of the current project is the development of a speech emotion recognition application to be also deployable at the edge using Intel's OpenVINO Toolkit. 


## Project's strategy:

The main steps followed, from pre-processing to final results, can be summarized as follows:
1. Initially some preliminary research on similar studies in the field, was conducted. Relative studies can be found in the Literature & Resources section.
2. An important step of the project is the selection of an appropriate dataset. At this step the RAVDESS datasets were selected for the study.
3. A pre-processing step was the conversion of the selected dataset from audio to spectrogram. At this step, it is also important to perform a clean up of the data, whenever necessary.
4. The research performed on similar studies, enabled us to identify possible gaps in this research field and address and solve these issues to our project. In our case, in order our application to be supported with Intel's OpenVINO Toolkit, it was necessary to first project audio recordigns into spectrogram representations, so as to enable deep learning with CNN instead of RNN-LSTM architecture(s)
5. Our model(s) were trained, based on couple different NN architectures, followed by a comparison and refinement of the models. 
6. Pytorch model was exported to ONNX format in order to be compatible with OpenVINO Toolkit Model Optimizer.
6. Optimization of the model in order to use quantization, fusing, and freezing by running the Model Optimizer, was also implemented. 
7. A conversion to Intermediate Representations with OpenVINO Toolkit, was performed.
8. The final steps include the developement of the user application and the deployment of the application.
9. Finally, testing of the application was performed in order to be certain about its effectiveness. 

In sections to follow, a detailed description of the processing strategy, as well as some key contributions of group members, will be included. 

## Original Datasets: 

The original datasets, that were selected for the current project are the Ravdess
1. [Ravdess Emotional Speech Audio Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)
2. [Ravdess Emotional Song Audio Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-song-audio)

The first dataset [1] contains  1440 files: 60 trials per actor x 24 actors = 1440. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

The second dataset [2] contains 1012 files: 44 trials per actor x 23 actors = 1012. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Song emotions includes calm, happy, sad, angry, and fearful expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.


## Pre-processed Datasets: 

The above two datasets, were pre-processed appropriately by George Christopoulos, before being further used in the model training. For this purpose, LibROSA python package (https://librosa.github.io/librosa/index.html) for music and audio analysis, was employed. This package provides the building blocks necessary to create music information retrieval systems. In our project, this package was employed in order to perform predictions of emotional types, in vocals of 5-seconds duration. These are the final datasets used for the emotion recognition project:

1. http://www.kaggle.com/dataset/2a4541d696a3fd076152467ace40a7bfe6e85e108f17292df04a4e6d7b4aecaa
2. http://www.kaggle.com/dataset/4e97bf2fb36d96647422054bfe9e0bdd34397120fb38eaf8f87c20f243acd511

### Spectograms examples
Some spectograms examples are provided:

1. Angry
![angry](https://github.com/geochri/SER/blob/master/images/03-02-05-01-01-01-05.jpg)
2. Calm
![Calm](https://github.com/geochri/SER/blob/master/images/03-02-02-01-01-02-19.jpg)
3. Disgust
![Disgust](https://github.com/geochri/SER/blob/master/images/03-01-07-02-01-02-21.jpg)
4. Fearful
![Fearful](https://github.com/geochri/SER/blob/master/images/03-01-06-02-01-02-07.jpg)
5. Happy
![Happy](https://github.com/geochri/SER/blob/master/images/03-02-03-01-02-01-15.jpg)
6. Neutral
![Neutral](https://github.com/geochri/SER/blob/master/images/speech_03-01-01-01-02-01-24.jpg)
7. Sad
![Sad](https://github.com/geochri/SER/blob/master/images/speech_03-01-04-01-02-02-08.jpg)
8. Surprised
![Surprised](https://github.com/geochri/SER/blob/master/images/speech_03-01-08-02-02-02-19.jpg)

## Model training and deployment

In this section the results of the two models with the best accuracies, also compatible with OpenVINO Toolkit Model Optimizer, will be presented.


### Model by Arka Chakraborty 

Pre-app model training results with all eight emotion types: 
```
Test Loss: 0.139551
Test Accuracy of angry: 100% (36/36)
Test Accuracy of  calm: 89% (33/37)
Test Accuracy of disgust: 100% (19/19)
Test Accuracy of fearful: 100% (37/37)
Test Accuracy of happy: 97% (36/37)
Test Accuracy of neutral: 88% (16/18)
Test Accuracy of   sad: 97% (36/37)
Test Accuracy of surprised: 100% (19/19)
Test Accuracy (Overall): 96% (232/240)
```
Dataset split:80(train sample)
              10(test sample)
              10(valid sample) 

Final results achieved: 
```
Test Loss: 0.058974
Test Accuracy of angry: 98% (75/76)
Test Accuracy of  calm: 97% (74/76)
Test Accuracy of disgust: 97% (38/39)
Test Accuracy of fearful: 97% (74/76)
Test Accuracy of happy: 98% (75/76)
Test Accuracy of neutral: 100% (67/67)
Test Accuracy of   sad: 98% (66/67)
Test Accuracy of surprised: 98% (66/67)
Test Accuracy (Overall): 98% (535/544)
```
The model missclassifies only nine images in a total of over 544 images.

Sample prediction: 
```
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Actual Emotion-sad
Predicted Emotion-sad
```
![Sad spectrogram 1](https://github.com/Speech-VINO/SER/blob/master/arka_sad_spectrogram1.png)

### Deployed Appplication using OpenVino:
![Deployed Appplication using OpenVino](https://github.com/Speech-VINO/SER/blob/master/deployment_openvino.png)

#### Repo:
https://github.com/Escanor1996/Speech-Emotion-Recognition-SER-


## Model by George Christopoulos

### Pre-Processing of audio to spectrogram and waveform:
#### kaggle links
[link1](http://www.kaggle.com/dataset/2a4541d696a3fd076152467ace40a7bfe6e85e108f17292df04a4e6d7b4aecaa)
[link2](http://www.kaggle.com/dataset/4e97bf2fb36d96647422054bfe9e0bdd34397120fb38eaf8f87c20f243acd511)

### Models

Experimented with merging all negative emotions into one: 
```
Test Loss: 0.552089
Test Accuracy of  calm: 80% (98/121)
Test Accuracy of happy: 68% (77/112)
Test Accuracy of negative: 95% (366/385)
Test Accuracy of neutral: 74% (44/59)
Test Accuracy of surprised: 84% (49/58)
Test Accuracy (Overall): 86.258503% (634/735)
```
Dataset split:60(train sample)
              20(test sample)
              20(valid sample) 

Final results achieved: 

```
Test Loss: 0.589566
Test Accuracy of  calm: 80% (89/110)
Test Accuracy of happy: 72% (82/113)
Test Accuracy of negative: 91% (372/406)
Test Accuracy of neutral: 78% (40/51)
Test Accuracy of surprised: 85% (47/55)
Test Accuracy (Overall): 85.714286% (630/735)
```
Dataset split:50(train sample)
              20(test sample)
              30(valid sample)  

Final and optimal results achieved for Ravdess dataset: 
```
CUDA is available!  Testing on GPU ...
Test Loss: 0.091140
Test Accuracy of angry: 97% (74/76)
Test Accuracy of  calm: 98% (82/83)
Test Accuracy of disgust: 97% (46/47)
Test Accuracy of fearful: 97% (68/70)
Test Accuracy of happy: 98% (65/66)
Test Accuracy of neutral: 100% (40/40)
Test Accuracy of   sad: 100% (70/70)
Test Accuracy of surprised: 97% (37/38)
Test Accuracy (Overall): 98.367347% (482/490)
```

### DesktopApp detection for both files wav and spectrogram image
![DesktopApp-AlphaVersion](https://github.com/geochri/SER/blob/master/desktopApp.png)

### Simple EdgeApp-DesktopApp demonstration videos
1. [EdgeApp](https://github.com/geochri/SER/blob/master/SER_project_edgeApp.mp4)
2. [DesktopApp](https://github.com/geochri/SER/blob/master/desktopApp_video.mp4)

#### Repo:
https://github.com/geochri/SER/



## Pytorch model exported to ONNX format

This simple example of convertion from pytorch to onnx was followed: https://michhar.github.io/convert-pytorch-onnx/

Example:

```
import torch
import torch.onnx

# A model class instance (class not shown)
model = MyModelClass()

# Load the weights from a file (.pth usually)
state_dict = torch.load(weights_path)

# Load the weights now into a model net architecture defined by our class
model.load_state_dict(state_dict)

# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(sample_batch_size, channel, height, width)
torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")
```
No extra dependencies needed, since pytorch itself provides us the library: 
https://pytorch.org/docs/stable/onnx.html#supported-operators



## Flask-WebApp - created by Arka

[Webapp](https://github.com/Escanor1996/Speech-Emotion-Recognition-SER-/tree/master/flask_webapp)



## Literature & Resources:

1. Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.
2. Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.
3. [Arxiv article: "Nonparallel Emotional Speech Conversion"](https://arxiv.org/abs/1811.01174)
4. [PLoS ONE article: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)
5. [Papers with Code: Speech Emotion Recognition articles](https://paperswithcode.com/task/speech-emotion-recognition)
6. [Arxiv article: Attention Based Fully Convolutional Network for Speech Emotion Recognition](https://arxiv.org/pdf/1806.01506.pdf)
7. [INTERSPEECH 2019 article: Self-attention for Speech Emotion Recognition](http://publications.idiap.ch/downloads/papers/2019/Tarantino_INTERSPEECH_2019.pdf)



## Contributors
@George Christopoulos - https://github.com/geochri

@Arka - https://github.com/Escanor1996

@K.S. - https://github.com/kakirastern

@Avinash Kumar - https://github.com/Avinashshah099

@Aarthi Alagammai - https://github.com/AarthiAlagammai

@Temitope Oladokun - https://github.com/TemitopeOladokun


