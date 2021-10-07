# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:07:05 2021

@author: user
"""

%matplotlib inline
import os
import pathlib
import tensorflow as tf
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR') 

# GPU 設定為 記憶體動態調整 (dynamic memory allocation)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# 下載模型，並解壓縮
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    # 解壓縮
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200711'
MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)
PATH_TO_MODEL_DIR

# 讀取 PATH_TO_MODEL_DIR 目錄下所有目錄及檔案
from os import listdir
from os.path import isfile, join
from os import path

for f in listdir(PATH_TO_MODEL_DIR):
    print(f)
# 快速從下載的目錄載入模型
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"

print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

# @tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

image_np = np.array(Image.open('./images_2/image2.jpg'))

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))


print(f'物件個數：{num_detections}')
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
print(f'物件資訊 (候選框, 類別, 機率)：')
for detection_boxes, detection_classes, detection_scores in \
    zip(detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']):
    print(np.around(detection_boxes,4), int(detection_classes)+1, round(detection_scores*100, 2))

detections['num_detections'] = num_detections
# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


# 將物件框起來
# min_score_thresh=.30 表機率(Confidence)至少要大於 30%
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes']+1,
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.30,
      agnostic_mode=False)

plt.figure(figsize=(12,8))
plt.imshow(image_np_with_detections, cmap='viridis')
plt.show()



# 存檔
plt.savefig('./images_2/detection3.png')
# plt.show()
from IPython.display import Image
Image('./images_2/detection3.png')