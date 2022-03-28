sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl libbz2-dev
curl -O https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tar.xz
tar -xf Python-3.7.3.tar.xz
cd Python-3.7.3
./configure --with-ssl
make
sudo make altinstall
which python3.7
rm -rf /usr/bin/python3
sudo ln -s /usr/local/bin/python3.7 /usr/bin/python3
which pip3.7
rm -rf /usr/bin/pip3
sudo ln -s /usr/local/bin/pip3.7 /usr/bin/pip3
