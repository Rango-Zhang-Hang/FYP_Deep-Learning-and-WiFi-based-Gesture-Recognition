wget https://raw.githubusercontent.com/zeroby0/nexmon_csi/pi-5.4.51-plus/install.sh -O install.sh
tmux new -c /home/pi -s nexmon 'bash install.sh | tee output.log'