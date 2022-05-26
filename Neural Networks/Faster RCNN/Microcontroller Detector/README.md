# Microcontroller Detector using TF Object Detection API

### 1. Set up tensorflow object detection API and pretrained model
### 2. Train the model
### 3. Test Data from URL

```python
!pip uninstall tensorflow
!pip install tensorflow==2.8
!pip uninstall opencv-python-headless
!pip install opencv-python-headless==4.1.2.30
!apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
```
> Make sure you installed tensorflow 2.6 or upper and opencv-python-headless 4.1.2.30

```python
import os
if not os.path.exists(os.getcwd() + 'models'):
  !git clone --depth 1 https://github.com/tensorflow/models
```
```python
# Install the Object Detection API
os.chdir('./models/research/')
!protoc object_detection/protos/*.proto --python_out=.
!cp object_detection/packages/tf2/setup.py .
!python -m pip install .
```
```python
!python ./object_detection/builders/model_builder_tf2_test.py
```
If you're trying to access kaggle dataset you need to add this command provided in google colab.
```python
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
# Then move kaggle.json into the folder where the API expects to find it.
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```
### Then, go to [kaggle.com](https://kaggle.com/) and click your profile from the right.
### Account>API> **click Create New API Token** button. JSON file will be download to your computer.

> Upload this kaggle json file

```python
!kaggle datasets download -d tannergi/microcontroller-detection
!unzip microcontroller-detection.zip
```
### Unzip the dataset from kaggle.

```python
!mv "Microcontroller Detection" microcontroller-detection
```
### Change the name of dataset.

```python
!wget https://raw.githubusercontent.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model/master/generate_tfrecord.py
!wget https://raw.githubusercontent.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model/master/training/labelmap.pbtxt
```
### Download required files. You can also get it from tensorflow github.

```python
!python generate_tfrecord.py --csv_input=microcontroller-detection/train_labels.csv --image_dir=microcontroller-detection/train --output_path=train.record
!python generate_tfrecord.py --csv_input=microcontroller-detection/test_labels.csv --image_dir=microcontroller-detection/test --output_path=test.record
```
### Set the path where you want to train your data.

```python
train_record_path = './train.record'
test_record_path = './test.record'
labelmap_path = './labelmap.pbtxt'
batch_size = 8
num_steps = 8000
num_eval_steps = 1000
```
### Some Variables.

```python
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
!tar -xf efficientdet_d0_coco17_tpu-32.tar.gz
fine_tune_checkpoint = 'efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0'
```
### Download base model you are going to train.

```python
!wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config

base_config_path = 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config'
```
### To customize efficientdet, you need it's config file. So we download config file from [Tensorflow Config](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2)

```python
import re

with open(base_config_path) as f:
    config = f.read()

with open('model_config.config', 'w') as f:
  
  # Set labelmap path
  config = re.sub('label_map_path: ".*?"', 
             'label_map_path: "{}"'.format(labelmap_path), config)
  
  # Set fine_tune_checkpoint path
  config = re.sub('fine_tune_checkpoint: ".*?"',
                  'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)
  
  # Set train tf-record file path
  config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 
                  'input_path: "{}"'.format(train_record_path), config)
  
  # Set test tf-record file path
  config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 
                  'input_path: "{}"'.format(test_record_path), config)
  
  # Set number of classes.
  config = re.sub('num_classes: [0-9]+',
                  'num_classes: {}'.format(4), config)
  
  # Set batch size
  config = re.sub('batch_size: [0-9]+',
                  'batch_size: {}'.format(batch_size), config)
  
  # Set training steps
  config = re.sub('num_steps: [0-9]+',
                  'num_steps: {}'.format(num_steps), config)
  
  # Set fine-tune checkpoint type to detection
  config = re.sub('fine_tune_checkpoint_type: "classification"', 
             'fine_tune_checkpoint_type: "{}"'.format('detection'), config)
  
  f.write(config)
```
### Edit the config file

```python
%cat model_config.config
```
### Let's check edited config file with command `%cat`

```python
model_dir = 'training/'
pipeline_config_path = './model_config.config'
!dir
```
### Check our current path.

```python
!python /content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_config_path} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={num_eval_steps}
```
### Let's Train the detector.
> It might take **1hr** to train this pretrained model using `batch size = 8`. More batchsize more memory need.

```python
%load_ext tensorboard
%tensorboard --logdir 'training/train'
```
### Load the tensorboard to check training records.

```python
output_directory = 'inference_graph'

!python /content/models/research/object_detection/exporter_main_v2.py \
    --trained_checkpoint_dir {model_dir} \
    --output_directory {output_directory} \
    --pipeline_config_path {pipeline_config_path}
```
### Save graph at inference.

```python
files.download(f'{output_directory}/saved_model/saved_model.pb')
```
### Download Trained model.

```python
import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

%matplotlib inline
```
### Let's test the model.

```python
def load_image_into_numpy_array(path):
  
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
```

```python
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
```

```python
tf.keras.backend.clear_session()
model = tf.saved_model.load(f'{output_directory}/saved_model')
```

```python
def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict
```

```python
for image_path in glob.glob('microcontroller-detection/test/*.jpg'):
  image_np = load_image_into_numpy_array(image_path)
  output_dict = run_inference_for_single_image(model, image_np)
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)
  display(Image.fromarray(image_np))
```
### You can see the result. 
### As a bonus, I will show you u can test data from url.
> only jpg format!!

```python
# custom image urls from internet

image_urls = ['https://www.electronicproducts.com/wp-content/uploads/board-level-products-communication-boards-arduino-spring-1.jpg',
              'https://static.generation-robots.com/12784-product_cover/official-raspberry-pi-3-b-starter-kit.jpg',
              'https://cdn.shopify.com/s/files/1/0300/6424/6919/articles/IoT-based-Accident-Detection-System.jpg?v=1592632076',
              'https://thingpulse.com/wp-content/uploads/2018/01/ComponentsWoodBg.jpg',
              'https://2.bp.blogspot.com/-eJut3cOhaMQ/XLHNYgX3xZI/AAAAAAAAMek/6qX6nh8WxCk3Ha8XwUdkjl5pbVPqzcIqgCLcBGAs/s1600/esp8266_blink_ssl.jpg',]
```

```python
from six.moves.urllib.request import urlopen
from six import BytesIO
from PIL import Image, ImageColor, ImageDraw , ImageFont , ImageOps
from IPython.display import display, Markdown ,clear_output
import tempfile
```

```python
for index, url in enumerate(image_urls):
  try:
    response = urlopen(url)
    img = response.read()
    img = BytesIO(img)
    pil_image = Image.open(img)
    pil_image = ImageOps.fit(pil_image, (512, 512), Image.ANTIALIAS)
    pil_image_rgb= pil_image.convert("RGB")
    _, filename = tempfile.mkstemp(suffix=".jpg")
    pil_image_rgb.save(filename, format="JPEG", quality=100)
    image_np = load_image_into_numpy_array(filename)
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)
    display(Image.fromarray(image_np))
  except:
    pass
``` 

### Let's go.. 
