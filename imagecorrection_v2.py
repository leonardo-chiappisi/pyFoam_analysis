
import csv
import matplotlib.colors as mcolors
from PIL import Image
import pandas as pd
import cv2   #install opencv-python
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import statistics



path_data = 'U:\\Uni\\PSCM Coordinator\\HDR\plots\\Foam\\data_analysis\\10 - SDS 2.5_1\\image'
path_result = 'U:\\Uni\\PSCM Coordinator\\HDR\plots\\Foam\\data_analysis\\10 - SDS 2.5_1\\image\\corrected'

for filename in sorted(os.listdir(path_data)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = Image.open(os.path.join(path_data, filename))
        width, height = image.size
        image = image.resize((int(width * np.sqrt(2)), height))
        image.save(os.path.join(path_result, "resized_" + filename))


    

  