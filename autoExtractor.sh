sleep 1s
str=`mcp -c 36/80 -C 1 -N 1`
sleep 1s
ifconfig wlan0 up
sleep 1s
nexutil -Iwlan0 -s500 -b -l34 -v$str
sleep 1s
iw dev wlan0 interface add mon0 type monitor
sleep 1s
ip link set mon0 up
sleep 1s
tcpdump -i wlan0 dst port 5500 -vv -w output.pcap -c 200