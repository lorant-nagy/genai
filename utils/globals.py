# globals.py
import os
DEVICE = os.environ.get("DEVICE", "cpu")   # "cuda" to use GPU
DTYPE = os.environ.get("DTYPE", "float32")  