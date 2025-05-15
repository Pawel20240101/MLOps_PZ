#!/bin/bash

mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 &

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root