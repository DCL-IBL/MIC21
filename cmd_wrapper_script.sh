#!/bin/bash

cd /host/mic21-framework/fiftyone
bash install.bash -d

python3 /host/mic21-framework/fiftyone/fiftyone/server/main.py &

sleep 100
python3 /host/mic21-framework/server/app.py &

cd /
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''
