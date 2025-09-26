# Log Anomaly Detection

This folder contains the **machine learning workload** for detecting anomalies in HDFS logs.  
It shows how to process large log datasets (>500k rows), tokenize log messages into sequences, and apply deep learning (CNN) for anomaly detection.

---

## Dataset
- **Source:** Public HDFS log dataset (~150 MB).  
- Due to size limits, **sample subsets** are included here: [`hdfs-log-anomaly-detection/sample_subset_data`](./sample_subset_data)  
- **Full dataset instructions:** You can download the complete HDFS log dataset from the [LogHub repository](https://github.com/logpai/loghub) (look for the **HDFS** logs).

---

## Methods
- **Preprocessing:** Clean log text, tokenize into integer indices, and apply padding for equal sequence length.  
- **Embeddings:** Learned embeddings using PyTorch's embedding layer.  
- **Model:** Convolutional Neural Network (CNN) with convolution, pooling, and fully connected layers for binary classification.  
- **Task:** Detect anomalous vs. normal log sequences.  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC.

---

## Tech Stack
- **Python:** pandas, numpy, matplotlib  
- **PyTorch:** for deep learning (embedding + CNN)  
- **Jupyter / Google Colab:** for interactive development and training

---

## Files
- `log_anomaly_detection_cnn.ipynb` : Main notebook with preprocessing + modeling.  
- `requirements.txt` : Python dependencies.  
- `hdfs_log_sample_data.csv` : Small demo dataset.  

---

## Usage

#### Install dependencies
pip install -r requirements.txt

#### Run notebook
jupyter notebook log_anomaly_detection_cnn.ipynb


#### Results

**Dataset imbalance:** test positive rate ≈ **0.952** (majority class).  
(Positive = "Normal", Negative = "Anomaly")

**CNN (PyTorch) - representative run**

- Test accuracy: **0.956**
- Precision / Recall / F1 @ threshold 0.5: **0.973 / 0.982 / 0.977**
- Confusion matrix @ 0.5: **[[55, 66], [44, 2335]]**  (TN, FP, FN, TP)
- Specificity @ 0.5: **0.455**
- ROC–AUC (test): **0.699** (≈ 0.692–0.699 across epochs)
- PR–AUC (positive / majority class): **0.968**
- PR–AUC (negative / minority class): **≈ 0.294**
- Best threshold (≈ **0.87–0.89**): **F1 remains 0.977** (Precision/Recall ≈ 0.973/0.981)

> Note: Because the dataset is highly imbalanced (~95% "Normal"), accuracy can be high even with low specificity. We therefore report PR–AUC for the **minority (anomaly)** class and include the full confusion matrix and specificity.
