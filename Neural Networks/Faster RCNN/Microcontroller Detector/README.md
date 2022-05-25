# Microcontroller Detector using TF Object Detection API

1. Set up tensorflow object detection API and pretrained model
2. Train the model
3. Test Data from URL

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
