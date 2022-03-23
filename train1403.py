import numpy as np
import tensorflow as tf
import random as python_random

from scipy.io import loadmat
from tensorflow.python import keras

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Dropout, MaxPool2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

#================================== dataSave first
import time
import numpy
import importlib
import config
from plotters.AmpPhaPlotter import Plotter # Amplitude and Phase plotter
decoder = importlib.import_module(f'mydecoders.{config.decoder}') # This is also an import

#4 gestures 2000 packets 84 subcarrier
a = numpy.zeros(shape=(1600,5,256),dtype=complex)
b = numpy.zeros(shape=(1600), dtype = "object") #'8', 'onefinger', 'five', 'C'
c = numpy.zeros(shape=(2000,1,256),dtype=complex)

pcap_filename1 = 'pcapfiles/eight2103.pcap'
pcap_filename2 = 'pcapfiles/six2103.pcap'
pcap_filename3 = 'pcapfiles/five2103.pcap'
pcap_filename4 = 'pcapfiles/C2103.pcap'
#....many as we want
index = 4
sample1 = decoder.read_pcap(pcap_filename1)
sample2 = decoder.read_pcap(pcap_filename2)
sample3 = decoder.read_pcap(pcap_filename3)
sample4 = decoder.read_pcap(pcap_filename4)

for j in range(400):
    for i in range(5):
        #packets 0-1999
        a[j, i] = sample1.get_csi((j+1)*(i+1)-1)
        b[j] = 'Eight'
        a[j+400, i] = sample2.get_csi((j+1)*(i+1)-1)
        b[j+400] = 'One'
        a[j+800, i] = sample3.get_csi((j+1)*(i+1)-1)
        b[j+800] = 'Five'
        a[j+1200, i] = sample4.get_csi((j+1)*(i+1)-1)
        b[j+1200] = 'C'


#================================== training
# Import dataset
#dataset = np.load('pcapfiles/dataset1403.npy',allow_pickle='TRUE')
#print(dataset)
# reshape CSI values
print(a.shape)
csi = a.reshape(1600,5,256,1)
print(csi.shape)


csi_abs = np.abs(csi)
csi_ang = np.angle(csi)
csi_tensor = np.concatenate((csi_abs,csi_ang),2) #256->512

csi_tensor /= np.max(csi_tensor)

#5 packet for one gesture collect: 1600,5,512
# Encode labels
label = b
encoder = LabelEncoder()
encoder.fit(label.ravel())
encoded_Y = encoder.transform(label.ravel())
dummy_y = to_categorical(encoded_Y)

#print(csi_tensor.shape)
#print(dummy_y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
     csi_tensor, dummy_y, test_size=0.2, random_state=73)
X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)

print(X_train.shape)
print(X_test.shape)


# Hyperparameters
numKernel = 16
kernelSize = (5,5)
poolSize = (3,3)
optimizer = optimizers.SGD()
epochs = 20
batchSize = 5
numClasses = 4
#iterataion = datasize/batchsize *epochs

# fix random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
python_random.seed(42)

#print(csi_tensor.shape)

csi_tensor = csi_tensor.reshape(1600,5,512,1) #batch, height, width, channels

# Create model
def baseline_model():
    model = Sequential()
    model.add(Conv2D(numKernel, kernelSize, padding='same',
                     input_shape=csi_tensor.shape[1:]
                     ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=poolSize))
    model.add(Flatten())
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])
    return model
model = baseline_model()

# Build the model and training
estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs,batch_size= batchSize, verbose=1)

history = estimator.fit(X_train, y_train,
                        validation_data = (X_test,y_test))



# 'less than samples'fold cross validation
kfold = KFold(n_splits=5, shuffle= True, random_state= 42)
crossval = cross_val_score(estimator, csi_tensor, dummy_y, cv = kfold)
print(crossval.mean())

model.save('FYPmodel')

# List all data in history
#print(history.history.keys())
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
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


































