# Multimodal Video Sentiment Analysis

This repository presents a late fusion multimodal approach to video sentiment analysis, combining audio and text data for enhanced sentiment classification.

## Prerequisites
- FFmpeg (version 6.1.1 or later) installed on your system.
- CUDA strongly recommended.

## Datasets

### Audio Dataset
- **[Download Audio Dataset](https://shorturl.at/bqIT1)**
  - **audio_embeddings.zip**: Contains audio embeddings for the dataset (suggested).
  - **final_v2_dataframe.csv**: CSV file containing the directory of audio files and their sentiment.
  - **Common.zip**: Audio files listed in final_v2_dataframe.csv.

### Text Dataset
- **[Download Text Dataset](https://shorturl.at/mptz7)**
  - **text.csv**: CSV file containing post-filtering text data for sentiment analysis.
  - **text_embeddings.zip**: Contains text embeddings for the dataset (suggested).

### Image Dataset
- **[Download Image Dataset](https://shorturl.at/uwBNS)**
  - **image_new.csv**: CSV file containing the self-curated image dataset.
  - **imaged.zip**: Contains all the images listed in image_new.csv.
  - **image_embeddings.zip**: Contains image embeddings for the dataset (suggested).

## Models Weights
- **[Download Model Weights](https://shorturl.at/klHLM)**
  - **audio_classifier.pth**: For audio classifier.
  - **new_image_classifier.pth**: For image classifier.
  - **text_classifier.pth**: For text classifier.

## Instructions
- Model architecture and training codes are included in `classifier.py`.
- An example of training is shown in `image_training_sample` notebook, using image classifier as an example.

### Order to Run
1. Download all datasets.
2. Optionally, use the models to generate embeddings, or use the provided embeddings to save time.
