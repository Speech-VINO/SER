# Speech Emotion Recognition Project for Intel Edge AI Scholarhip Challenge 2020 SpeechVINO study group

## Goal:
Develop a speech emotion recognition application to be deployable at the edge using Intel's OpenVINO Toolkit.

## Motivation:
Speech emotion recognition is interesting in its own right as some project of academic interest, because its application enables us to understand the range of human emotions better. We have tried to simplify the range of emotions during the training of the convolutional neural network to only a few by combining similar emotions, but have discovered that the best approach forward is to classify humans emotions into at least eight categories as follows: angry, calm, disgust, fearful, happy neutral, sad, surprised. Also, we wanted to test if such a classification task can be carried out by first turning our audio clips into some visual representations, in our case a spectrogram, before employing computer vision techniques for the task. This turns out to be possible and can even be optimized for deployment at the edge.

## Social Impact: 
Detection of human emotions can be tricky with voice samples alone. By turning the speech AI problem into a computer vision one, we have been able to achieve a level of accuracies (>95% across the board) that makes it easy even for machines to distinguish between different emotions in the human voice. This can help some people with problems understanding emotions to have an external and objective check, and to learn to distinguish between emotions with practice.

## Plan of Attack:
1. Do prelimiary research on similar work done in the area
2. Choose some appropriate dataset like the RAVDESS datasets for the study
3. Convert the dataset from audio to spectrogram (and try cleaning up if necessary) in the pre-processing step 
4. Find out what has not been tried before and then attack this gap of research accordingly: In our case, to first project audio recordigns into spectrogram representations so as to enable deep learning with CNN instead of RNN-LSTM architecture(s)
5. Train our model(s) based on couple different NN architectures, then compare and refine. 
6. Optimize the model for using quantization, fusing, and freezing by running the Model Optimizer. 
7. Convert to Intermediate Representations with OpenVINO Toolkit.
8. Develope user application.
9. Deployment of user application. 
10. Testing. 

## Outcome of the project 
As a result of the work of the members of our project, we have been able to come up with two applications using CNNs in the modeling to predict emotions from some input audio file. One is an edge app, the other one is a desktop app. 

## How to Use the Desktop App
1. Install the edge app on your local machine
2. Choose either a .wav file containing the audio sample or a .jpeg file containing the spectrogram converted from and is representing the audio sample of a certain emotion, which can be one of the following: angry, calm, disguist, fearful, happy, neutral, sad, and surprised. 
3. Enter the absolute path of the chosen file to the field in the GUI. 
4. Press the "Predict... " button on the right hand side of the GUI, to get the emotion predicted. 
5. Voila! You should be getting a predicted emotion within seconds. 

## How to Use the Edge App
Direction of usage: Using the terminal, entering a command starting with `python app_ser.py ...` to run the application, and with some flags such as `-i` to specify the input file, `-t` to specify type such as "IMG", then `-m` to specify the model, and finally `-c` to specify the extension, an inference at the edge can be made. 

## Work by Arka
Pre-app model training results with all 8 emotion types: 
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
Used split 80-10-10 for train-valid-test. 

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
It is only miss classifying 9 images over 544 images.

Sample prediction: 
```
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Actual Emotion-sad
Predicted Emotion-sad
```
![Sad spectrogram 1](https://github.com/Speech-VINO/SER/blob/master/arka_sad_spectrogram1.png)

## Deployed Appplication using OpenVino:
![Deployed Appplication using OpenVino](https://github.com/Speech-VINO/SER/blob/master/deployment_openvino.png)

## Repo:
https://github.com/Escanor1996/Speech-Emotion-Recognition-SER-

To convert pytorch to onnx: https://michhar.github.io/convert-pytorch-onnx/ (Not very complicated)
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

## Work by George
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
With a 60/20/20 split. 

```
Test Loss: 0.589566
Test Accuracy of  calm: 80% (89/110)
Test Accuracy of happy: 72% (82/113)
Test Accuracy of negative: 91% (372/406)
Test Accuracy of neutral: 78% (40/51)
Test Accuracy of surprised: 85% (47/55)
Test Accuracy (Overall): 85.714286% (630/735)
```
With a 50/20/30 split. 

Final results achieved: 
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
Ravdess challenge: solved.

## DesktopApp detection for both files wav and spectrogram image
![DesktopApp-AlphaVersion](https://github.com/geochri/SER/blob/master/desktopApp.png)

## Simple EdgeApp-DesktopApp demonstration videos
1. [EdgeApp](https://github.com/geochri/SER/blob/master/SER_project_edgeApp.mp4) (Note: This demo video has to be viewed with VLC or other compatible software)
2. [DesktopApp](https://github.com/geochri/SER/blob/master/desktopApp_video.mp4)


## Pre-Processing of audio to spectrogram and waveform:
https://www.kaggle.com/timolee/audio-data-conversion-to-images-eda

## Original Datasets: 
1. [Ravdess Emotional Speech Audio Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)
2. [Ravdess Emotional Song Audio Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-song-audio)

## Pre-processed Datasets: 
1. http://www.kaggle.com/dataset/2a4541d696a3fd076152467ace40a7bfe6e85e108f17292df04a4e6d7b4aecaa
2. http://www.kaggle.com/dataset/4e97bf2fb36d96647422054bfe9e0bdd34397120fb38eaf8f87c20f243acd511

## Literature & Resources:
1. [Medium article: "Speech Emotion Recognition with CNNs"](https://towardsdatascience.com/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3)
2. [Arxiv article: "Nonparallel Emotional Speech Conversion"](https://arxiv.org/abs/1811.01174)
3. [PLoS ONE article: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)
4. [Papers with Code: Speech Emotion Recognition articles](https://paperswithcode.com/task/speech-emotion-recognition)
5. [Medium article: AEI: Artificial 'Emotional' Intelligence](https://towardsdatascience.com/aei-artificial-emotional-intelligence-ea3667d8ece)
6. [Arxiv article: Attention Based Fully Convolutional Network forSpeech Emotion Recognition](https://arxiv.org/pdf/1806.01506.pdf)
7. [INTERSPEECH 2019 article: Self-attention for Speech Emotion Recognition](http://publications.idiap.ch/downloads/papers/2019/Tarantino_INTERSPEECH_2019.pdf)
8. https://github.com/marcogdepinto/Emotion-Classification-Ravdess

## Sample Repos Serves as an Example for Workflow: 
https://github.com/meshaun9/openvino_speech_recognition
* We will need to specify the hardware and software requirements for the setup
* We could use either a `.py` or `.ipynb` file for organizing and running our code
