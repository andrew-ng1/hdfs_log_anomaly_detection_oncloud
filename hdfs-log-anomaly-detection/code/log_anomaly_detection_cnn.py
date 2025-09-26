"""
# HDFS Logs Anomaly Detection using Convolutional Neural Networks (CNN) - Andrew Nguyen
"""

import pandas as pd
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from torch import nn
import torch
import math
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
nltk.download('punkt')
nltk.download('punkt_tab')

from google.colab import drive
drive.mount('/content/drive')

label = pd.read_csv('/content/drive/MyDrive/anomaly_label.csv')

label

label['Label'][:10000].value_counts()

import pandas as pd
import json
import re

path='/content/drive/MyDrive/HDFS.log'

log_data=open(path,'r')
pattern = r'[0-9]'
regex1 = 'blk_\d+'
regex2 = 'blk_-\d+'
data = []
for line in log_data:
    # Match all digits in the string and replace them by empty string
    l = []
    blk_id = re.findall(regex1, line)
    try:
      l.append(blk_id[0])
      mod_string = re.sub(pattern, '', line)
      mod_string  = mod_string.replace('INFO','').lower().strip()
      l.append(mod_string)
      data.append(l)
    except:
      blk_id = re.findall(regex2, line)
      try:
        l.append(blk_id[0])
        mod_string = re.sub(pattern, '', line)
        mod_string  = mod_string.replace('INFO','').lower().strip()
        l.append(mod_string)
        data.append(l)
      except:
        print(line)

df=pd.DataFrame(data,columns=['BlockId','content'])

df.head()

df.tail()

final_data = df.merge(label,on='BlockId',how='left')

final_data.head()

def change_label(row):
  if row['Label']=="Normal":
    return 1
  return 0

final_data['Label'] = final_data.apply(change_label,axis=1)

final_data.head()

final_data.to_csv("final_data.csv")

class Preprocessing:
  def __init__(self, num_words, seq_len):
    self.data = 'final_data.csv'
    self.num_words = num_words
    self.seq_len = seq_len
    self.vocabulary = None
    self.x_tokenized = None
    self.x_padded = None
    self.x_raw = None
    self.y = None

    self.x_train = None
    self.x_test = None
    self.y_train = None
    self.y_test = None

  def load_data(self):
    # Reads the raw csv file and split into
    # sentences (x) and target (y)
    df = pd.read_csv(self.data)[:10000]
    self.x_raw = df['content'].values
    self.y = df['Label'].values

  def clean_text(self):
    # Removes special symbols and just keep
    # words in lower or upper form
    self.x_raw = [x.lower() for x in self.x_raw]
    self.x_raw = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.x_raw]

  def text_tokenization(self):
    # Tokenizes each sentence by implementing the nltk tool
    self.x_raw = [word_tokenize(x) for x in self.x_raw]

  def build_vocabulary(self):
    # Builds the vocabulary and keeps the "x" most frequent word
    self.vocabulary = dict()
    fdist = nltk.FreqDist()

    for sentence in self.x_raw:
      for word in sentence:
        fdist[word] += 1

    common_words = fdist.most_common(self.num_words)

    for idx, word in enumerate(common_words):
      self.vocabulary[word[0]] = (idx+1)

  def word_to_idx(self):
    # By using the dictionary (vocabulary), it is transformed
    # each token into its index based representatio
    self.x_tokenized = list()

    for sentence in self.x_raw:
      temp_sentence = list()
      for word in sentence:
        if word in self.vocabulary.keys():
          temp_sentence.append(self.vocabulary[word])
      self.x_tokenized.append(temp_sentence)

  def padding_sentences(self):
    # Each sentence which does not fulfill the required le
    # it's padded with the index 0
    pad_idx = 0
    self.x_padded = list()

    for sentence in self.x_tokenized:
      while len(sentence) < self.seq_len:
        sentence.insert(len(sentence), pad_idx)
      self.x_padded.append(sentence)

    self.x_padded = np.array(self.x_padded)

  def split_data(self):
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_padded, self.y, test_size=0.25, random_state=42)

class TextClassifier(nn.ModuleList):

	def __init__(self, params):
		super(TextClassifier, self).__init__()

		# Parameters regarding text preprocessing
		self.seq_len = params.seq_len
		self.num_words = params.num_words
		self.embedding_size = params.embedding_size

		# Dropout definition
		self.dropout = nn.Dropout(0.25)

		# CNN parameters definition
		# Kernel sizes
		self.kernel_1 = 2
		self.kernel_2 = 3
		self.kernel_3 = 4
		self.kernel_4 = 5

		# Output size for each convolution
		self.out_size = params.out_size
		# Number of strides for each convolution
		self.stride = params.stride

		# Embedding layer definition # is used to convert the data into vector
		self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0)

		# Convolution layers definition
		self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
		self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
		self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
		self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)

		# Max pooling layers definition
		self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
		self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
		self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
		self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

		# Fully connected layer definition
		self.fc = nn.Linear(self.in_features_fc(), 1)


	def in_features_fc(self):
		# Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
		out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
		out_conv_1 = math.floor(out_conv_1)
		out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
		out_pool_1 = math.floor(out_pool_1)

		# Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
		out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
		out_conv_2 = math.floor(out_conv_2)
		out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
		out_pool_2 = math.floor(out_pool_2)

		# Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
		out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
		out_conv_3 = math.floor(out_conv_3)
		out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
		out_pool_3 = math.floor(out_pool_3)

		# Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
		out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
		out_conv_4 = math.floor(out_conv_4)
		out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
		out_pool_4 = math.floor(out_pool_4)

		# Returns "flattened" vector (input for fully connected layer)
		return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size



	def forward(self, x):

		# Sequence of tokes is filterd through an embedding layer
		x = self.embedding(x)

		# Convolution layer 1 is applied
		x1 = self.conv_1(x)
		x1 = torch.relu(x1)
		x1 = self.pool_1(x1)

		# Convolution layer 2 is applied
		x2 = self.conv_2(x)
		x2 = torch.relu((x2))
		x2 = self.pool_2(x2)

		# Convolution layer 3 is applied
		x3 = self.conv_3(x)
		x3 = torch.relu(x3)
		x3 = self.pool_3(x3)

		# Convolution layer 4 is applied
		x4 = self.conv_4(x)
		x4 = torch.relu(x4)
		x4 = self.pool_4(x4)

		# The output of each convolutional layer is concatenated into a unique vector
		union = torch.cat((x1, x2, x3, x4), 2)
		union = union.reshape(union.size(0), -1)

		# The "flattened" vector is passed through a fully connected layer
		out = self.fc(union)
		# Dropout is applied
		out = self.dropout(out)
		# Activation function is applied
		out = torch.sigmoid(out)

		return out.squeeze()

from torch.utils.data import Dataset, DataLoader

class DatasetMaper(Dataset):

   def __init__(self, x, y):
      self.x = x
      self.y = y

   def __len__(self):
      return len(self.x)

   def __getitem__(self, idx):
      return self.x[idx], self.y[idx]

class Parameters:
   # Preprocessing parameeters
   seq_len: int = 35
   num_words: int = 2000

   # Model parameters
   embedding_size: int = 64
   out_size: int = 32
   stride: int = 2

   # Training parameters
   epochs: int = 10
   batch_size: int = 12
   learning_rate: float = 0.001

def calculate_accuray(grand_truth, predictions):
		# Metrics calculation
		true_positives = 0
		true_negatives = 0
		for true, pred in zip(grand_truth, predictions):
			if (pred >= 0.5) and (true == 1):
				true_positives += 1
			elif (pred < 0.5) and (true == 0):
				true_negatives += 1
			else:
				pass
		# Return accuracy
		return (true_positives+true_negatives) / len(grand_truth)

def evaluation(model, loader_test):
  # Set the model in evaluation mode
  model.eval()
  predictions = []
  # Starst evaluation phase
  try:
    with torch.no_grad():
      for x_batch, y_batch in loader_test:
        y_pred = model(x_batch)
        predictions += list(y_pred.detach().numpy())
  except:
    pass
  return predictions

def prepare_data(num_words, seq_len):
		# Preprocessing pipeline
		pr = Preprocessing(num_words, seq_len)
		pr.load_data()
		pr.clean_text()
		pr.text_tokenization()
		pr.build_vocabulary()
		pr.word_to_idx()
		pr.padding_sentences()
		pr.split_data()

		return {'x_train': pr.x_train, 'y_train': pr.y_train, 'x_test': pr.x_test, 'y_test': pr.y_test}

def train(model, data, params):

   # Initialize dataset maper
   train = DatasetMaper(data['x_train'], data['y_train'])
   test = DatasetMaper(data['x_test'], data['y_test'])

   # Initialize loaders
   loader_train = DataLoader(train, batch_size=params.batch_size)
   loader_test = DataLoader(test, batch_size=params.batch_size)

   # Define optimizer
   optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)

   # Starts training phase
   for epoch in range(params.epochs):
      # Set model in training model
      model.train()
      predictions = []
      # Starts batch training
      for x_batch, y_batch in loader_train:

         y_batch = y_batch.type(torch.FloatTensor)

         # Feed the model
         y_pred = model(x_batch)

         # Loss calculation
         loss = F.binary_cross_entropy(y_pred, y_batch)

         # Clean gradientes
         optimizer.zero_grad()

         # Gradients calculation
         loss.backward()

         # Gradients update
         optimizer.step()

         # Save predictions
         predictions += list(y_pred.detach().numpy())

      # Evaluation phase
      test_predictions = evaluation(model, loader_test)

      # Metrics calculation
      train_accuary = calculate_accuray(data['y_train'], predictions)
      test_accuracy = calculate_accuray(data['y_test'], test_predictions)
      print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuary, test_accuracy))
      y_train_true = np.asarray(data['y_train']).ravel()
      y_test_true  = np.asarray(data['y_test']).ravel()
      y_train_prob = np.asarray(predictions).ravel()
      y_test_prob  = np.asarray(test_predictions).ravel()

      def safe_auc(y_true, y_score):
         # AUC requires both classes present
         return roc_auc_score(y_true, y_score) if np.unique(y_true).size == 2 else float("nan")

      train_auc = safe_auc(y_train_true, y_train_prob)
      test_auc  = safe_auc(y_test_true,  y_test_prob)

      if np.isnan(train_auc) or np.isnan(test_auc):
         print("          AUC: n/a (single class present)")
      else:
         print("          Train AUC: %.5f, Test AUC: %.5f" % (train_auc, test_auc))

      from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, average_precision_score

      # --- extra: test metrics at 0.5 and best threshold ---
      y_test_true = np.asarray(data['y_test']).ravel()
      y_test_prob = np.asarray(test_predictions).ravel()

      # @0.5 threshold (what most readers expect)
      y_hat_05 = (y_test_prob >= 0.5).astype(int)
      prec, rec, f1, _ = precision_recall_fscore_support(y_test_true, y_hat_05, average="binary", zero_division=0)
      cm = confusion_matrix(y_test_true, y_hat_05)
      print(f"          @0.5 -> Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  Confusion: {cm.tolist()}")

      # Best threshold by Youden’s J (maximizes TPR - FPR)
      fpr, tpr, thr = roc_curve(y_test_true, y_test_prob)
      best_idx = np.argmax(tpr - fpr)
      best_thr = float(thr[best_idx])
      y_hat_best = (y_test_prob >= best_thr).astype(int)
      prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(y_test_true, y_hat_best, average="binary", zero_division=0)
      print(f"          @best thr={best_thr:.3f} -> Precision: {prec_b:.3f}  Recall: {rec_b:.3f}  F1: {f1_b:.3f}")

      # (Optional) PR-AUC (good when positives are rare)
      ap = average_precision_score(y_test_true, y_test_prob)
      print(f"          PR AUC: {ap:.4f}")

      # imbalance & ranges
      print("          Pos rate (train/test):",
            float(np.mean(data['y_train'])), float(np.mean(data['y_test'])))

      # negative-class PR-AUC (since negatives are the minority)
      from sklearn.metrics import average_precision_score, confusion_matrix
      y_true = np.asarray(data['y_test']).ravel()
      y_prob = np.asarray(test_predictions).ravel()
      neg_pr_auc = average_precision_score(1 - y_true, 1 - y_prob)
      print(f"          PR AUC (negative/minority): {neg_pr_auc:.4f}")

      # specificity at 0.5 (how well we catch negatives)
      cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
      tn, fp, fn, tp = cm.ravel()
      spec = tn / (tn + fp) if (tn + fp) else float('nan')
      print(f"          Specificity @0.5: {spec:.3f}")

para = Parameters()

import numpy as np
from sklearn.metrics import roc_auc_score

data = prepare_data(para.num_words, para.seq_len)
model = TextClassifier(para)
train(model, data, para)

