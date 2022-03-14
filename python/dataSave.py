import time
import numpy
import importlib
import config
from plotters.AmpPhaPlotter import Plotter # Amplitude and Phase plotter
decoder = importlib.import_module(f'mydecoders.{config.decoder}') # This is also an import

#4 gestures 2000 packets 84 subcarrier
a = numpy.zeros(shape=(4,2000,256),dtype=complex)
b = ['8', 'onefinger', 'five', 'C']

pcap_filename1 = 'pcapfiles/eight1403.pcap'
pcap_filename2 = 'pcapfiles/midfinger1403.pcap'
pcap_filename3 = 'pcapfiles/wave1403.pcap'
pcap_filename4 = 'pcapfiles/C_1103_2000.pcap'
#....many as we want
index = 4
sample1 = decoder.read_pcap(pcap_filename1)
sample2 = decoder.read_pcap(pcap_filename2)
sample3 = decoder.read_pcap(pcap_filename3)
sample4 = decoder.read_pcap(pcap_filename4)


for i in range(2000):
    #packets 0-1999
    a[0, i] = sample1.get_csi(i)
    a[1, i] = sample2.get_csi(i)
    a[2, i] = sample3.get_csi(i)
    a[3, i] = sample4.get_csi(i)

d = {'csi' : a, 'label' : b }

numpy.save('pcapfiles/dataset1403.npy', d)