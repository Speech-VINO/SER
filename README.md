# Speech Emotion Recognition Project for Intel Edge AI Scholarhip Challenge 2020 SpeechVINO study group

## Goal:
Develop a speech emotion recognition application to be deployable at the edge using Intel's OpenVINO Toolkit.

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


Repo:https://github.com/Escanor1996/Speech-Emotion-Recognition-SER-

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
