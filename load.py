import numpy as np
import tensorflow as tf
import random as python_random

#load trained model
loadpath = "./model"
model = keras.models.load_model(loadpath, compile = True)

dataset= /***
***
**/

# Extract and reshape CSI values
csi = dataset['csid_lab']
csi_abs = np.abs(csi)
csi_ang = np.angle(csi)
csi_tensor = np.concatenate((csi_abs,csi_ang),1)
csi_tensor = np.swapaxes(csi_tensor,0,3)
csi_tensor = np.swapaxes(csi_tensor,1,3)
csi_tensor = np.swapaxes(csi_tensor,2,3)
csi_tensor /= np.max(csi_tensor)

test_input;

# Generate predictions for samples--softmax output
predictions = model.predict(samples_to_predict)
print(predictions)

# Generate arg maxes for predictions
classes = np.argmax(predictions, axis = 1)
print(classes)