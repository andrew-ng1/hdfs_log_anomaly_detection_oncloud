# Cloud-Based Data Pipeline with Log Anomaly Detection

This repository demonstrates how to build and run an end-to-end data workflow on the cloud.  
It combines two parts:
1. **Cloud Infrastructure Workflow (AWS EC2 + S3 + CLI):**  showing how to manage cloud resources, move data between local and cloud, and run analysis remotely.
2. **Log Anomaly Detection (Applied ML Workload):** applying a CNN-based deep learning ML model to detect anomalies in large-scale HDFS logs (>500k rows).


By combining these, the project highlights how to design **scalable, cloud-enabled analytics pipelines** that can support both simple analyses and more advanced machine learning tasks.

---

## Highlights
- **Cloud Pipeline**: Start, stop, and terminate EC2 instances via AWS CLI.  
- **Data Management**: Upload/download datasets between local and S3.  
- **Remote Execution**: Run Python workloads on EC2 and retrieve results.  
- **Anomaly Detection**: Apply a CNN-based deep learning ML models to detect unusual patterns in HDFS logs (>500k rows).  
- **Scalability**: Demonstrates how local analytics can be extended to cloud environments.  

---

## Results
- **HDFS Log Anomaly Detection (CNN, PyTorch):** achieved ~95.6% test accuracy and **F1 = 0.977**.  
- Full metrics (confusion matrix, PR–AUC for anomalies, threshold tuning) are provided in the  
  [`hdfs-log-anomaly-detection`](./hdfs-log-anomaly-detection) subfolder.

---

## Tech Stack
- **Cloud**: AWS EC2, S3, CLI, Linux, SCP  
- **Languages**: Python (pandas, numpy, matplotlib, scikit-learn)  
- **Tools**: Jupyter/Colab, Power BI (optional for visualization)  

---

## Usage
1. See [`aws-pipeline-demo`](./aws-pipeline-demo) for step-by-step instructions on building the cloud workflow.  
2. See [`hdfs-log-anomaly-detection`](./hdsf-log-anomaly-detection) for applying anomaly detection to log data.  
3. Combine both to create a **cloud-based anomaly detection pipeline**.

---