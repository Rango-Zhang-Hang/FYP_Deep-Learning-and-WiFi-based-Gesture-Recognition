## FYP_Hand Gesture Recognition Based on WI-FI Signals and Deep Learning
This project allows me to extract Channel State Information(CSI) and use Convolutional Nerual Network(CNN) to recognize these data frame to related hand gestures labels.
* [Get_started](#get_started)
* [CSI_Measurement](#csi_measurement)
* [RaspberryPi_setup](#raspberrypi_setup)
* [Technologies](#technologies)

For full information about CSI, please check [NEXMON_CSI](https://github.com/seemoo-lab/nexmon_csi).

## Get_started
Buy a Raspberry Pi4.
	
## CSI_measurement  
First, finish the [setup_guides](#setup) and make sure the [versions](#technologies) are correct.

1. Use **mcp** to generate a parameter string which is used to configure the extractor,you can also check the full conmmand list with **mcp -h**. The following sample  collects    CSI on channel 36 (5GHz, it is the only channel worked in most of the time, and also 5GHz has less interruptions) with bandwidth 80 MHz on first core of the WiFi chip, for the  first spatial stream. Raspberry Pi has only one core, and a single antenna, so the last two options don't need changing.
   ```
   mcp -c 157/80 -C 1 -N 1
   ```
   Or you can type
   ```
   mcp -c 157/80 -C 1 -N 1 -b $MAC 0X80	
   ```
   where the **$MAC** is the router amc address.
   Also, 
   ```
   str='mcp -c 157/80 -C 1 -N 1'
   ```
   is accpeted.
   
2. Run the following lines:
   ```
   ifconfig wlan0 up
   nexutil -Iwlan0 -s500 -b -l34 -v$str
   iw dev wlan0 interface add mon0 type monitor
   ip link set mon0 up
   ```
3. Collect the CSI by listening on socket 5500 for UDP packets. One way to do this is using tcpdump: ```tcpdump -i wlan0 dst port 5500```. You can store 1000 CSI samples in a pcap file like this: ```tcpdump -i wlan0 dst port 5500 -vv -w output.pcap -c 1000```.

## RaspberryPi_setup
For full information, move to [zero0-4.19.97](https://github.com/nexmonster/nexmon_csi/tree/pi-4.19.97)
Install with commands:

```
$unfinshed
```

## Technologies
Project is created with:
* Kernel version: 5.4
* Image version: 2020-08-20-buster-lite
* Firmware version: 7_45_189 
* WI-FI chip: bcm43455c0
* Device used: Raspberry Pi B4
	



