# Multimodal Deepfake Image Detection System

This repository contains a PyTorch-based implementation of a multimodal deepfake detection system for a Nigerian university undergraduate project. The system uses a hybrid CNN-Transformer-BERT model to detect deepfake images in cyber threats, trained on multimodal datasets like DFDC and FakeAVCeleb, following a design-implement-evaluate approach. It generates metrics such as accuracy, precision, recall, and F1-score, saved as `deepfake_detection_results.csv`.

## Environment Setup

### Prerequisites
- **OS**: Linux/Unix (e.g., Ubuntu, Google Colab); Windows/Mac compatible with adjustments.
- **Python**: 3.8 or higher.
- **Hardware**: CPU or GPU (GPU recommended for faster training).
- **Storage**: ~5 GB for datasets, model checkpoints, and outputs.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MultimodalDeepfakeDetection.git
   cd MultimodalDeepfakeDetection
   ```
2. Install dependencies:
   ```bash
   pip install torch==1.10.0 transformers==4.35.0 datasets==2.14.0 opencv-python==4.6.0 numpy==1.24.3 pandas==2.0.3 pillow==9.5.0
   ```
3. Verify installation:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

## Dataset Structure

### DFDC and FakeAVCeleb Datasets

- **Path**: `data/DFDC/` and `data/FakeAVCeleb/`.
- **Structure**:
  ```
  data/
  ├── DFDC/
  │   ├── train/
  │   │   ├── real/*.mp4
  │   │   └── fake/*.mp4
  │   ├── test/
  │   │   ├── real/*.mp4
  │   │   └── fake/*.mp4
  │   └── valid/
  │       ├── real/*.mp4
  │       └── fake/*.mp4
  ├── FakeAVCeleb/
  │   ├── train/
  │   │   ├── real/*.mp4
  │   │   └── fake/*.mp4
  │   ├── test/
  │   │   ├── real/*.mp4
  │   │   └── fake/*.mp4
  │   └── valid/
  │       ├── real/*.mp4
  │       └── fake/*.mp4
  ```
- **Description**: Contains `.mp4` videos (real: authentic, fake: deepfake). Each split (`train`, `test`, `valid`) has `real` and `fake` subfolders.
- **Acquisition**: 
  - [DFDC dataset](https://ai.facebook.com/datasets/dfdc/)
  - [FakeAVCeleb dataset](https://github.com/DASH-Lab/FakeAVCeleb)

## Run Instructions

1. **Prepare Dataset**:
   - Download and place DFDC and FakeAVCeleb datasets in `data/DFDC/` and `data/FakeAVCeleb/`.
   - If using Google Colab, upload datasets to `/content/drive/MyDrive/datasets/` and adjust paths in the code.
   - Verify directory structure as shown above.

2. **Run the Code**:
   - Save the provided code as `scripts/deepfake_detection.py`.
   - Execute:
     ```bash
     cd scripts
     python deepfake_detection.py
     ```
   - **Note**: Adjust `dataset_dir` in `main()` if your dataset path differs (e.g., `data/DFDC` or `data/FakeAVCeleb`).

3. **Execution Details**:
   - Loads and preprocesses video frames (resized to 224x224, normalized).
   - Processes audio and text metadata using Transformer and BERT models.
   - Trains the hybrid model (10 epochs, batch size 8).
   - Saves model checkpoints to `checkpoints/`.
   - Computes and saves metrics to `results/deepfake_detection_results.csv`.

## Output Paths

- **Results**: `results/deepfake_detection_results.csv`
  - Contains: Accuracy, Precision, Recall, F1_Score.
  - Example:
    ```
    Accuracy,Precision,Recall,F1_Score
    0.92,0.90,0.93,0.91
    ```
- **Model Checkpoints**: `checkpoints/deepfake_model_{epoch}-{val_loss}.pth`
  - Best models based on validation loss.
- **Logs**: Console output includes dataset loading status, model training progress, and metric values.

## Metrics

The code evaluates the hybrid model with:

- **Classification Metrics**:
  - **Accuracy**: Proportion of correct predictions (0–1, higher is better).
  - **Precision**: Proportion of true positive detections (0–1, higher is better).
  - **Recall**: Proportion of actual positives detected (0–1, higher is better).
  - **F1-Score**: Harmonic mean of precision and recall (0–1, higher is better).
- **Output**: Metrics are printed to the console and saved in `results/deepfake_detection_results.csv`.

## Directory Structure

```
MultimodalDeepfakeDetection/
├── data/
│   ├── DFDC/
│   │   ├── train/
│   │   ├── test/
│   │   └── valid/
│   └── FakeAVCeleb/
│       ├── train/
│       ├── test/
│       └── valid/
├── scripts/
│   └── deepfake_detection.py
├── checkpoints/
│   └── deepfake_model_*.pth
├── results/
│   └── deepfake_detection_results.csv
└── README.md
```

## Notes

- **Dataset Access**: If DFDC or FakeAVCeleb is unavailable, consider using smaller subsets or synthetic datasets for testing.
- **Dataset Maturity**: Output/metrics may vary based on dataset size and quality. Use larger, diverse datasets for improved metrics.
- **Troubleshooting**:
  - **No videos loaded**: Verify dataset paths and `.mp4` files. Check directory structure.
  - **Shape errors**: Ensure video frames are resized correctly (e.g., 224x224). Adjust preprocessing if needed.
  - **Memory issues**: Reduce batch size (e.g., 4) or use CPU if GPU memory is limited.
- **License**: MIT License
