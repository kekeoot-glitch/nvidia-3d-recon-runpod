#!/bin/bash
set -e

# Start SSH service
echo "Starting SSH server..."
service ssh start
/usr/sbin/sshd -D &

# Start Jupyter Notebook
echo "Starting Jupyter Lab..."
cd /app/3d-object-reconstruction
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'
