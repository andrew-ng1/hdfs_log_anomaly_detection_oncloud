# AWS Pipeline Demo

This folder demonstrates a **simple cloud workflow** using AWS EC2 + S3.
The example workload is a Python script (`test.py`) that reads a CSV, makes a plot, and writes output.jpg.

---

## Workflow Overview
1. **Provision EC2**: Start an instance via AWS CLI.  
2. **Create S3 Bucket**: Store and retrieve datasets.  
3. **Upload Dataset**: Send local files to S3.  
4. **Run Analysis on EC2**: Download dataset from S3, run Python script (`test.py`).  
5. **Export Results**: Save output image and transfer back to local machine via `scp`.  
6. **Shutdown**: Stop or terminate instance to manage costs.

---

## 🛠️ Tech Used
- AWS EC2, S3, CLI  
- Linux shell, SCP  
- Python (pandas, numpy, matplotlib)  

---

## 📂 Files
- `test.py` - Example workload: scatterplot + linear regression line.  
- `/screenshots/` - Step-by-step screenshots (create bucket, upload dataset, run analysis, transfer output).  

---

## 🔧 Example Commands

1. **Launch / Start / Stop EC2**
##### Launch a new instance
aws ec2 run-instances --image-id ami-0a09ba4ff4bac79c6 --instance-type t2.micro

##### Get instance IDs
aws ec2 describe-instances --query "Reservations[].Instances[].InstanceId"

##### Check instance status
aws ec2 describe-instance-status --instance-id i-01f26e09f640c8c02

##### Start existing instance
aws ec2 start-instances --instance-ids i-01f26e09f640c8c02

##### Stop instance
aws ec2 stop-instances --instance-ids i-01f26e09f640c8c02

2. **Create S3 bucket**
aws s3api create-bucket --bucket lab3-andrew-bucket --region us-east-1

3. **Upload dataset & script from local to S3**
aws s3 cp /mnt/c/Users/16478/cities-job-satisfaction.csv s3://lab3-andrew-bucket/
aws s3 cp /mnt/c/Users/16478/test.py s3://lab3-andrew-bucket/

##### (Optional) List objects in the bucket
aws s3 ls s3://lab3-andrew-bucket

4. **On the EC2 box: download from S3**
aws s3 cp s3://lab3-andrew-bucket/test.py .
aws s3 cp s3://lab3-andrew-bucket/cities-job-satisfaction.csv .

5. **Run the analysis on EC2**
python3 test.py
##### (you’ll see a preview of the CSV; file 'output.jpg' is created)
ls

6. **Copy result back to local**
sudo scp -i "/mnt/c/Users/16478/Downloads/AndrewDATA534.pem" \
  ubuntu@ec2-52-23-209-198.compute-1.amazonaws.com:~/output.jpg \
  /mnt/c/Users/16478/Desktop/DATA534

7. **Terminate when done**
aws ec2 terminate-instances --instance-ids i-01f26e09f640c8c02
