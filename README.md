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

### Videos for Inference
- **[Download Results (CSV)](https://shorturl.at/asGNP)**
  - Note: The last character of the video file name indicates the sentiment. For instance, video_0 refers to negative, and video_1 refers to positive.
  - Due to large file sizes, video data has been split into several zip files:
    - **video.zip**
    - **extra_vid.zip**
    - **extra_vid2.zip**

## Models Weights
- **[Download Model Weights](https://shorturl.at/klHLM)**
  - **audio_classifier.pth**: For audio classifier.
  - **new_image_classifier.pth**: For image classifier.
  - **text_classifier.pth**: For text classifier.

## Evaluation Files
- **[Download Results (CSV)](https://shorturl.at/xGJT4)**
  - **final_result.csv**: List of results for all individual modalities of the test videos.

## Instructions
- Model architecture and training codes are included in `classifier.py`.
- An example of training is shown in `image_training_sample` notebook, using the image classifier as an example.
- The integrated models are computationally expensive to run together due to resource consumption. Hence, the notebooks have split the models by modality, as seen in `inference_sample`.
- Evaluation results of individual classifiers can be concatenated using Python or Excel. Final evaluation results are shown in `final_result.csv`.

### Model Training
1. Download all datasets.
2. Optionally, use the models to generate embeddings, or use the provided embeddings to save time.
3. Refer to the sample notebook training. An example of the image classifier is provided in `image_training_sample`.

### Model Inference and Testing on Unseen Videos
1. Load the models and use them for inference:

    ```python
    checkpoint_path = './new_image_classifier.pth'
    loaded_checkpoint = torch.load(checkpoint_path)
    image_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    image_model.eval()

    checkpoint_path = './text_classifier.pth'
    loaded_checkpoint = torch.load(checkpoint_path)
    text_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    text_model.eval()

    checkpoint_path = './audio_classifier.pth'
    loaded_checkpoint = torch.load(checkpoint_path)
    audio_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    audio_model.eval()
    ```

2. Refer to `inference_sample` for an example.

### Voting Mechanism and Evaluation
1. Use `final_result.csv` that has already been processed.
2. Run the sample notebook given, titled `final_evaluation`.
