"""Simple logger module for VASA project."""

import logging
import sys
import os

# Suppress MediaPipe's verbose C++ logs (GPU initialization spam)
# These logs appear when MediaPipe initializes GPU context for face mesh processing
os.environ['GLOG_minloglevel'] = '3'  # Suppress INFO, WARNING, ERROR from glog (MediaPipe C++)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'  # Keep GPU enabled, just suppress logs

# Suppress protobuf warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Protobuf.*')

# Create logger
logger = logging.getLogger('vasa')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

# Add handler
if not logger.handlers:
    logger.addHandler(console_handler)
