#!/bin/bash

set -e

cd ./OPS

python stocks.py

python preprocessing.py

python Feature_pipeline.py

python Feature_view.py

python Training_pipeline.py

python Inference_pipeline.py