import numpy as np
import tensorflow as tf
import random as python_random

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, Activation, Flatten, Dense
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Dropout, MaxPool2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

# fix random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
python_random.seed(42)

# Import dataset
dataset = loadmat('dataset_lab_276_dl.mat')

# Extract and reshape CSI values
csi = dataset['csid_lab']
csi_abs = np.abs(csi)
csi_ang = np.angle(csi)
csi_tensor = np.concatenate((csi_abs,csi_ang),1)
csi_tensor = np.swapaxes(csi_tensor,0,3)
csi_tensor = np.swapaxes(csi_tensor,1,3)
csi_tensor = np.swapaxes(csi_tensor,2,3)
csi_tensor /= np.max(csi_tensor)

# Encode labels
label = dataset['label_lab']
encoder = LabelEncoder()
encoder.fit(label.ravel())
encoded_Y = encoder.transform(label.ravel())
dummy_y = to_categorical(encoded_Y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
     csi_tensor, dummy_y, test_size=0.2, random_state=73)
X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)

# Hyperparameters
numKernel = 16
kernelSize = (5,5)
poolSize = (3,3)
optimizer = optimizers.SGD()
epochs = 10
batchSize = 10
numClasses = 276

# Create model
def CNNLSTM():
    model = Sequential()
    model.add(ConvLSTM2D(8, (3,3),
                     input_shape=(200,60,3,1),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3,3)))
    model.add(Flatten())
    model.add(Dense(180))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer= 'SGD',
                  metrics=['accuracy'])
    return model
model = CNNLSTM()

# Build the model and training
estimator = KerasClassifier(build_fn=CNNLSTM, epochs=epochs,batch_size= batchSize, verbose=1)

history = estimator.fit(X_train, y_train,
                        validation_data = (X_test,y_test))



# 5-fold cross validation
kfold = KFold(n_splits=5, shuffle= True, random_state= 42)
crossval = cross_val_score(estimator, csi_tensor, dummy_y, cv = kfold)
print(crossval.mean())

# List all data in history
print(history.history.keys())
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()