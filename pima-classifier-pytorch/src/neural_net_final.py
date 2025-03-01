"""
Use this example - https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/ to understand pytorch NNs. Augment the dataset by perturbing the data fields with gaussian noise. Train/test/validate. Take a screencast walkthrough of the full code, including your understanding of how the NN is implemented.

1) Number of times pregnant 2) Plasma glucose concentration at 2 hours in an oral glucose tolerance test 3) Diastolic blood pressure (mm Hg) 4) Triceps skin fold thickness (mm) 5) 2-hour serum insulin (μIU/ml) 6) Body mass index (weight in kg/(height in m)2) 7) Diabetes pedigree function 8) Age (years)
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class PimaIndiansDiabetesClassifier:
  def __init__(self, dataset, startIdx, endIdx):
    self.X = dataset[:, startIdx:endIdx]
    self.y = dataset[:, endIdx]
    self.input_size = self.X.shape[1]
    self.output_size = 1

  def plot_outliers(self):
    for i in range(0,8):
      plt.figure(figsize=(6, 4))
      plt.boxplot(self.X[:,i])
      plt.show()
    
  def find_correlation(self,idx):
    for i in idx:
      correlation, _ = pearsonr(self.X[:,i], self.y)
      print(f"Correlation Coefficient : {correlation}")

      # sns.pairplot(pd.DataFrame(self.X))

      # plt.scatter(self.X[:,i], self.y)
      # plt.title('Scatter Plot of X and Y')
      # plt.xlabel('X')
      # plt.ylabel('Y')
      # plt.show()

      columns_to_drop = []
      if correlation < 0.09:
        columns_to_drop.append(i)
    
    self.X = np.delete(self.X, columns_to_drop, axis=1)
    self.input_size = self.input_size - len(columns_to_drop)

  def fill_zeros(self,idx,n):
    # feature = self.X[:,idx]
    # non_zero_feature_values = feature[feature != 0]
    # mean_feature = non_zero_feature_values.mean()
    # feature_imputed = np.where(feature == 0, mean_feature, feature)

    for i in idx:
      imputer = KNNImputer(n_neighbors=n)
      self.X[:,i] = np.where(self.X[:,i] == 0, np.nan ,self.X[:,i])
      self.X[:,i] = imputer.fit_transform(self.X[:,i].reshape(-1,1))[:,0]

  def scale(self, scaler="std"):
    if scaler == "std":
      self.scaler = StandardScaler()
    elif scaler == "rob":
      self.scaler = RobustScaler()
    elif scaler == "min":
      self.scaler = MinMaxScaler()

    self.X = self.scaler.fit_transform(self.X)

  def split(self, test_size=0.2, random_state=42):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=test_size,random_state=random_state)

  def add_noise(self, mean=0, std_dev=0.1, noisy_samples=0.2):
    self.size = (self.X_train.shape[0], self.X_train.shape[1])
    self.noise = np.random.normal(mean, std_dev, size=self.size)
    self.noisy_sample_size_train = int(noisy_samples * len(self.X_train))
    self.X_train_noisy = self.X_train + self.noise
    self.X_train = np.concatenate((self.X_train, self.X_train_noisy[:self.noisy_sample_size_train]), axis=0)
    self.y_train = np.concatenate((self.y_train, self.y_train[:self.noisy_sample_size_train ]),axis=0)

  def type_cast(self):
    self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
    self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
    self.y_train = torch.tensor(self.y_train, dtype=torch.float32).reshape(-1,1)
    self.y_test = torch.tensor(self.y_test, dtype=torch.float32).reshape(-1,1)

  def init_model(self,n1=12,n2=16,n3=8,alpha=0.0001):
    self.alpha = alpha
    # self.model = nn.Sequential(nn.Linear(self.input_size,n1), nn.ReLU(),nn.Linear(n1,n2), nn.ReLU(), nn.Linear(n2,n3), nn.ReLU(), nn.Linear(n3,self.output_size), nn.Sigmoid())
    self.model = nn.Sequential(nn.Linear(self.input_size,n1), nn.ReLU(), nn.Linear(n1,n3), nn.ReLU(), nn.Linear(n3,self.output_size), nn.Sigmoid())
    self.loss_fn = nn.BCELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

  def train(self, epochs=40, batch_size=11):
    self.epochs = epochs
    self.batch_size = batch_size
    self.train_loss = []
    self.train_acc = []

    for epoch in range(self.epochs):
      for i in range(0, len(self.X_train), self.batch_size):
        Xbatch = self.X_train[i:i + self.batch_size]
        y_pred = self.model(Xbatch)
        ybatch = self.y_train[i:i + self.batch_size]

        loss = self.loss_fn(y_pred, ybatch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      with torch.no_grad():
        y_pred = self.model(self.X_train)
        accuracy = (y_pred.round() == self.y_train).float().mean()
        self.train_acc.append(accuracy)

      self.train_loss.append(loss.item())

      print(f"Finished epoch: {epoch}, train loss: {loss} accuracy: {accuracy}")

  def plot_acc_loss(self):
    plt.figure(figsize=(8, 4))

    epo = [i for i in range(0, self.epochs)]

    plt.plot(epo, self.train_loss, color='y', label='Training Loss')
    plt.plot(epo, self.train_acc,   label='Training Accuracy')

    plt.title('Training Loss & Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

  def find_acc(self):
    with torch.no_grad():
      y_pred = self.model(self.X_test)

    self.accuracy = (y_pred.round() == self.y_test).float().mean()
    return self.accuracy


dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

pima = PimaIndiansDiabetesClassifier(dataset,0,8)
pima.find_correlation([0,1,2,3,4,5,6,7])
pima.fill_zeros([1, 2, 3, 4, 5],10)
pima.scale("rob")
pima.split(0.2,42)
pima.add_noise(0,0.1,0.9)
pima.type_cast()
pima.init_model(12,16,8,0.0001)
pima.train(40,10)
print(f"accuracy: {pima.find_acc()}")
