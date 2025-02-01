# Reservoir Computing Projects

This repository contains several projects based on Reservoir Computing (RC), with a focus on Echo State Networks (ESN). The projects address tasks such as anomaly detection in ECG signals, time-series prediction, and the NARMA benchmark task.

## Overview

Reservoir Computing models, particularly Echo State Networks (ESN), are used for time-series prediction and sequence modeling. This repository includes various tasks where ESNs are applied, including ECG anomaly detection, NARMA task solving, spoken digit recognition, and sunspot prediction. Each task is represented with its own script or notebook.

## Features

- Implementation of ESNs for different machine learning tasks.
- Configurable hyperparameters.
- Visualization of results.
- Support for various datasets.

## File Structure

C:
│  .gitignore
│  LICENSE
│  README.md
│  requirements.txt
│
├─config
│      esn_config_yaml
│
├─figures
│      delay1.png
│      delay2.png
│      ecg_anomaly_detection.png
│      narma.png
│      nonlinearity.png
│      spoken_digit_recognition.png
│      sunspot_prediction.png
│
├─notebooks
│  │  delay.ipynb
│  │  ecg_anomaly_detection.ipynb
│  │  esn.py
│  │  narma.ipynb
│  │  nonlinearity.ipynb
│  │  spoken_digit_recognition.ipynb
│  │  sunspot_prediction.ipynb
│  │
│  ├─.ipynb_checkpoints
│  │      delay-checkpoint.ipynb
│  │      ecg_anomaly_detection-checkpoint.ipynb
│  │      narma-checkpoint.ipynb
│  │      nonlinearity-checkpoint.ipynb
│  │      spoken_digit_recognition-checkpoint.ipynb
│  │      sunspot_prediction-checkpoint.ipynb
│  │
│  ├─data
│  │  │  anomaly.txt
│  │  │  normal.txt
│  │  │  SN_ms_tot_V2.0.txt
│  │  │
│  │  └─Lyon_decimation_128
│  │          s1_u10_d0.mat
│  │          s1_u10_d1.mat
│  │          s1_u10_d2.mat
│  │
│  └─__pycache__
│          esn.cpython-310.pyc
│
└─src
    │  delay.py
    │  ecg_anomaly_detection.py
    │  esn.py
    │  narma.py
    │  nonlinearity.py
    │  spoken_digit_recognition.py
    │  sunspot_prediction.py
    │  __init__.py
    │
    ├─data
    │  │  anomaly.txt
    │  │  normal.txt
    │  │  SN_ms_tot_V2.0.txt
    │  │
    │  └─Lyon_decimation_128
    │          s1_u10_d0.mat
    │          s1_u10_d1.mat
    │          s1_u10_d2.mat
    │
    └─__pycache__
            esn.cpython-39.pyc


## Setup

1. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

2. You can run each task using the Jupyter notebooks in the `notebooks` directory.

## Tasks and Implementations

### Echo State Network (ESN)

- **Purpose**: Implement an Echo State Network for the NARMA benchmark task.
- **Files**: 
    - `narma.py`: Code to solve the NARMA task using ESN.
    - `esn.py`: Core ESN implementation.
    - `esn_config_yaml`: Configuration file for ESN hyperparameters.

### ECG Anomaly Detection

- **Purpose**: Anomaly detection in ECG signals using ESNs.
- **Files**: 
    - `ecg_anomaly_detection.py`: Algorithm for anomaly detection in ECG data.
    - `ecg_anomaly_detection.ipynb`: Jupyter notebook for running the anomaly detection task.

### Spoken Digit Recognition

- **Purpose**: Implement a model for recognizing spoken digits.
- **Files**: 
    - `spoken_digit_recognition.py`: Script to train and test the spoken digit recognition model.
    - `spoken_digit_recognition.ipynb`: Notebook for testing and evaluating the recognition task.

### Sunspot Prediction

- **Purpose**: Predict sunspot activity using time-series prediction with ESN.
- **Files**: 
    - `sunspot_prediction.py`: Script for sunspot prediction using ESNs.
    - `sunspot_prediction.ipynb`: Jupyter notebook for evaluating the sunspot prediction task.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## References

1. **田中剛平**, **中根了昌**, and **廣瀬明**. *リザバーコンピューティング: 時系列パターン認識のための高速機械学習の理論とハードウェア*. 森北出版, 2021. ISBN 978-4-627-85531-1. [Amazon](https://www.amazon.co.jp/dp/4627855311)