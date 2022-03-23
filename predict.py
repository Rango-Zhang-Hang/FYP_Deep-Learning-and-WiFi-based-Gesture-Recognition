import time
import numpy as np
import importlib
import config
from tensorflow import keras
from plotters.AmpPhaPlotter import Plotter # Amplitude and Phase plotter
decoder = importlib.import_module(f'mydecoders.{config.decoder}') # This is also an import


label = ['Eight', 'OneFinger', 'Five', 'C']
model = keras.models.load_model('FYPmodel')

pcap_filename = 'pcapfiles/test.pcap'
sample = decoder.read_pcap(pcap_filename)
csi = np.zeros(shape=(1,5,256),dtype=complex)

for i in range(5):
    # packets 0-1999
    csi[0,i] = sample.get_csi(i)

csi = csi.reshape(1,5,256,1)
print(csi.shape)
csi_abs = np.abs(csi)
csi_ang = np.angle(csi)
csi_tensor = np.concatenate((csi_abs,csi_ang),2) #256->512

csi_tensor /= np.max(csi_tensor)


# Generate a prediction using model.predict()
# and calculate it's shape:
print("Generate a prediction...")
prediction = model.predict(csi_tensor)

max = 0.00000000
l = 0
for i in range(4):
    if max < prediction[0,i]:
        max = prediction[0,i]
        l = i
#for i in range prediction:

print("It should be: ", label[l], ' where ',
      prediction[0,0], ' to be ', label[0], ', ',
      prediction[0,1], ' to be ', label[1], ', ',
      prediction[0,2], ' to be ', label[2], ', ',
      prediction[0,3], ' to be ', label[3],)