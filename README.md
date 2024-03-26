# Multimodal Video Sentiment Analysis

This repository presents a late fusion multimodal approach to video sentiment analysis, combining audio and text data for enhanced sentiment classification.


Prerequisites
You must have FFmpeg (from 6.1.1) available on your system.
Cuda is strongly recommended

## Audio Dataset
You can download the audio dataset from the following link:
[Audio Dataset](https://shorturl.at/bqIT1)

### Contents:
- **audio_embeddings.zip**: Contains audio embeddings for the dataset (suggested)
- **final_v2_dataframe.csv**: CSV file containing the directory of the audio files and its sentiment
- **Common.zip**: Audio files listed in the final_v2_dataframe.csv

## Text Dataset
You can download the text dataset from the following link:
[Text Dataset](https://shorturl.at/mptz7)

### Contents:
- **text.csv**: CSV file containing post-filtering text data for sentiment analysis.
- **text_embeddings.zip**: Contains text embeddings for the dataset (suggested)

## Image Dataset
You can download the image dataset from the following link:
[Image Dataset](https://shorturl.at/uwBNS)

### Contents:
- **image_new.csv**: CSV file containing the self-curated image dataset
- **imaged.zip**: Contains all the images listed in the image_new.csv
- **image_embeddings.zip**: Contains image embeddings for the dataset (suggested)

## Models weights
You can download the image dataset from the following link:
[Link to model](https://shorturl.at/klHLM)

### Contents:
- **audio_classifier.pth**: For audio classifier
- **image_classifier.pth**: For image classifier
- **text_classifier.pth**: For text classifier


