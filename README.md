# Speech Emotion Recognition Project for Intel Edge AI Scholarhip Challenge 2020 SpeechVINO study group

## Goal:
Develop a speech emotion recognition application to be deployable at the edge using Intel's OpenVINO Toolkit.

## Plan of Attack:
1. Do prelimiary research on similar work done in the area
2. Find out what has not been tried before and then attack this gap of research accordingly: In our case, to first project audio recordigns into spetrogram representations so as to enable deep learning with CNN instead of RNN-LSTM architecture(s)
3. Train our model(s) based on couple different NN architectures, then compare and refine. 
4. Convert to Intermediate Representations with OpenVINO Toolkit.
The rest => TODO

## Pre-Processing of audio to spectrogram and waveform:
https://www.kaggle.com/timolee/audio-data-conversion-to-images-eda

## Datasets: 
1. [Ravdess Emotional Speeck Audio Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)
2. [Ravdess Emotional Song Audio Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-song-audio)

## Literature & Resources:
1. [Medium article: "Speech Emotion Recognition with CNNs"](https://towardsdatascience.com/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3)
2. [Arxiv article: "Nonparallel Emotional Speech Conversion"](https://arxiv.org/abs/1811.01174)
3. [PLoS ONE article: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)
4. [Papers with Code: Speech Emotion Recognition articles](https://paperswithcode.com/task/speech-emotion-recognition)
5. [Medium article: AEI: Artificial 'Emotional' Intelligence](https://towardsdatascience.com/aei-artificial-emotional-intelligence-ea3667d8ece)
