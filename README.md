# Reservoir Computing Projects

This repository contains several projects based on Reservoir Computing (RC), with a focus on Echo State Networks (ESN). The projects address tasks such as anomaly detection in ECG signals, time-series prediction, and the NARMA benchmark task.

## Overview

Reservoir Computing models, particularly Echo State Networks (ESN), are used for time-series prediction and sequence modeling. This repository includes various tasks where ESNs are applied, including ECG anomaly detection, NARMA task solving, spoken digit recognition, and sunspot prediction. Each task is represented with its own script or notebook.

## Features

- Implementation of ESNs for different machine learning tasks.
- Configurable hyperparameters.
- Visualization of results.
- Support for various datasets.


1. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

2. You can run each task using the Jupyter notebooks in the `notebooks` directory.

## Tasks and Implementations

### Echo State Network (ESN) - NARMA Benchmark

- **Purpose**: Implement an Echo State Network for solving the NARMA benchmark task, a standard test for evaluating nonlinear temporal processing.
- **Files**:  
  - `narma.py`: ESN implementation for the NARMA task.  
  - `esn.py`: Core ESN architecture and utilities.  
  - `esn_config_yaml`: YAML configuration for hyperparameters.

### ECG Anomaly Detection

- **Purpose**: Detect anomalies in electrocardiogram (ECG) signals using temporal dynamics captured by the ESN.
- **Files**:  
  - `ecg_anomaly_detection.py`: Main logic for detecting anomalies.  
  - `ecg_anomaly_detection.ipynb`: Interactive notebook for demonstration and analysis.

### Short-Term Memory (Delay Task)

- **Purpose**: Evaluate the short-term memory capacity of the ESN by reconstructing input signals after fixed delays.
- **Files**:  
  - `delay.py`: Code to run the delay memory task.  
  - `delay.ipynb`: Notebook visualization of memory performance.

### Nonlinearity Task

- **Purpose**: Test the ESN's ability to learn and represent nonlinear transformations of input signals.
- **Files**:  
  - `nonlinearity.py`: Code to model nonlinear input-output relationships.  
  - `nonlinearity.ipynb`: Notebook for experimentation.

### Spoken Digit Recognition

- **Purpose**: Classify spoken digits from audio inputs using reservoir dynamics.
- **Files**:  
  - `spoken_digit_recognition.py`: Preprocessing and classification pipeline.  
  - `spoken_digit_recognition.ipynb`: Evaluation and visualization notebook.

### Sunspot Time-Series Prediction

- **Purpose**: Forecast future sunspot activity based on historical time-series data.
- **Files**:  
  - `sunspot_prediction.py`: Time-series forecasting with ESN.  
  - `sunspot_prediction.ipynb`: Visualization and result analysis.


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## References

1. **田中剛平**, **中根了昌**, and **廣瀬明**. *リザバーコンピューティング: 時系列パターン認識のための高速機械学習の理論とハードウェア*. 森北出版, 2021. ISBN 978-4-627-85531-1. [Amazon](https://www.amazon.co.jp/dp/4627855311)
