import os
import time
import numpy as np
import importlib
import config
from tensorflow import keras
from plotters.AmpPhaPlotter import Plotter # Amplitude and Phase plotter
decoder = importlib.import_module(f'mydecoders.{config.decoder}') # This is also an import


label = ['Eight', 'OneFinger', 'Five', 'C', 'OK', 'Empty', 'Cover']
model = keras.models.load_model('FYPmodel')

print('test')


os.system("str=`mcp -c 36/80 -C 1 -N 1`")

os.system("ifconfig wlan0 up")
os.system("nexutil -Iwlan0 -s500 -b -l34 -v$str")
os.system("iw dev wlan0 interface add mon0 type monitor")
os.system("ip link set mon0 up")

cover = True
ready = False
a = True

while True:
    a = True
    #collect packet real-time
    print('Hold on...')
    time.sleep(1)
    os.system("tcpdump -i wlan0 dst port 5500 -vv -w output.pcap -c 5")
    print('Tcpdump 5 packets')

    pcap_filename = 'test.pcap'
    sample = decoder.read_pcap(pcap_filename)
    csi = np.zeros(shape=(1, 5, 256), dtype=complex)

    for i in range(5):
        # packets 0-1999
        csi[0, i] = sample.get_csi(i)

    csi = csi.reshape(1, 5, 256, 1)
    #print(csi.shape)
    csi_abs = np.abs(csi)
    csi_ang = np.angle(csi)
    csi_tensor = np.concatenate((csi_abs, csi_ang), 2)  # 256->512

    csi_tensor /= np.max(csi_tensor)

    # Generate a prediction using model.predict()
    # and calculate it's shape:
    #print("Generate a prediction...")
    prediction = model.predict(csi_tensor)

    max = 0.00000000
    l = 0
    for i in range(4):
        if max < prediction[0, i]:
            max = prediction[0, i]
            l = i
    # for i in range prediction:
    if label[l] == 'Cover':
        ready = True
        # predict activate
        print('Please ready to the position...')
        in_content = input('Press any key to continue: ')
        print('5...')
        time.sleep(1)
        print('4...')
        time.sleep(1)
        print('3...')
        time.sleep(1)
        print('2...')
        time.sleep(1)
        print('1...')
        time.sleep(1)
    elif (label[l]!='Cover') & ready:
        # the collection
        print('Pleasew wait...')
        print("It should be: ", label[l], ' where ',
              prediction[0, 0], ' to be ', label[0], ', ',
              prediction[0, 1], ' to be ', label[1], ', ',
              prediction[0, 2], ' to be ', label[2], ', ',
              prediction[0, 3], ' to be ', label[3], ', ',
              prediction[0, 4], ' to be ', label[4], ', ',
              prediction[0, 5], ' to be ', label[5], ', ',
              prediction[0, 6], ' to be ', label[6])
        while a:
            print('Do you want to continue? [y/n]')
            in_content = input('y/n: ')
            if (in_content == 'Y') | (in_content == 'y'):
                print('Continue...')
                a = False
            elif (in_content == 'N') | (in_content == 'n'):
                print('Exit')
                os.system("sleep 1s")
                exit(0)
            else:
                print("wrong")













