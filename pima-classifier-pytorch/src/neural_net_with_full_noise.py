import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

X = dataset[:, 0:8]
y = dataset[:, 8]

print(f"Size of X: {X.shape[0]}")

# Replace the zero values with mean (blood pressure & triceps skin fold thickness)

blood_pressure = X[:,2]
non_zero_blood_pressure_values = blood_pressure[blood_pressure !=0]
mean_blood_pressure = non_zero_blood_pressure_values.mean()

blood_pressure_imputed = np.where(blood_pressure == 0, mean_blood_pressure, blood_pressure)

X[:,2] = blood_pressure_imputed


tricep_skin_fold_thickness = X[:,3]
non_zero_tricep_skin_fold_thickness_values = tricep_skin_fold_thickness[tricep_skin_fold_thickness != 0]
mean_tricep_skin_fold_thickness = non_zero_tricep_skin_fold_thickness_values.mean()

tricep_skin_fold_thickness_imputed = np.where(tricep_skin_fold_thickness == 0,mean_tricep_skin_fold_thickness, tricep_skin_fold_thickness)

X[:,3] = tricep_skin_fold_thickness_imputed

# Normalize values

scaler = StandardScaler()

X_normalized = scaler.fit_transform(X)

# Adding guassian noise to the features 
mean = 0
std_dev = 0.1

noise = np.random.normal(mean, std_dev, size=(X_normalized.shape[0], X_normalized.shape[1]))

X_normalized = X_normalized + noise

X_temp, X_test, y_temp, y_test = train_test_split(X_normalized,y, test_size=0.2,random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

print(f"Size of X_train: {X_train.shape[0]}, Size of y_train: {y_train.shape[0]}")

# Adding noise to 20% of the data and appending it to the end of original data
#X_noisy = X_train + noise
#noisy_sample_size_for_train = int(0.5 * len(X_train))
#X_noisy_for_train = X_noisy[:noisy_sample_size_for_train]
#print(f"Size of X_noisy_for_train: {X_noisy_for_train.shape[0]}")
#X_train_augmented = np.concatenate((X_train, X_noisy_for_train), axis=0)
#y_train = np.concatenate((y_train , y_train[:noisy_sample_size_for_train]), axis=0)



# Convert the data to 32 bit float for compatability

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1,1)

print(f"Shape of X_train: {X_train.shape[0]}  {X_train.shape[1]}")

# Define model

model = nn.Sequential(nn.Linear(8,12), nn.ReLU(), nn.Linear(12,8), nn.ReLU(), nn.Linear(8,1), nn.Sigmoid())

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 200
batch_size = 10

# Train model

for epoch in range(n_epochs):
    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i + batch_size]
        y_pred = model(Xbatch)
        ybatch = y_train[i:i + batch_size]

        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss = 0.0
    with torch.no_grad():
        y_pred = model(X_val)
        loss = loss_fn(y_pred, y_val)
        val_loss += loss.item()

    avg_val_loss = val_loss / len(X_val)

    print(f"Finished epoch: {epoch}, train loss: {loss}, validation loss: {avg_val_loss}")



with torch.no_grad():
    y_pred = model(X_test)


accuracy = (y_pred.round() == y_test).float().mean()

print(f"Accuracy: {accuracy}") 
