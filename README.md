# NC-TF-OBJECT-DETECTION 
> This tutorial running with Ubuntu 16.04LTS, python 2.7


# PREPARE DATA

  - (1) Download training model: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  - (2) Download example: https://github.com/nhancv/nc-tf-object-detection
  - (3) Dowload Tensorflow models: https://github.com/tensorflow/models
    ```
    git clone https://github.com/tensorflow/models.git
    git checkout 4b8fe70416fe4826a3bad622e56780a7c2eb330c
    ```
 - Put all files from (1) + (2)/src + (2)/script to (3)/research/object_detection

# CONFIGURE
## Install tensorflow
> Replace `targetDirectory` to specific path of tensorflow where you want to installed.
1. Install pip, virtualenv
```
sudo apt-get install python-pip python-dev python-virtualenv
```
2. Config env path. Declare snip code below to `~/.bash_aliases`
```
#Tensorflow
export TF_CPP_MIN_LOG_LEVEL=2 
export TF_ROOT="/Volumes/Soft/_Program_Files/tensorflow"
alias tensorflow="$TF_ROOT/bin/python"
alias tensorflow.setup="virtualenv --system-site-packages $TF_ROOT && tensorflow.active && pip install --upgrade tensorflow"
alias tensorflow.destroy="rm -rf $TF_ROOT"
alias tensorflow.active="source $TF_ROOT/bin/activate"
alias tensorboard="$TF_ROOT/bin/tensorboard"
alias tensorboard.log="$TF_ROOT/bin/tensorboard --logdir=./logs"
```
3. Update with new bash_aliases content by restart terminal or just run `source ~/.bash_aliases`
4. Create tensorflow workspace
```
tensorflow.setup
```
5. Whenever you want working with tensorflow just active python evn first
```
$ tensorflow.active

# => (targetDirectory)$
```
6. If you finish your work
```
(targetDirectory)$ deactivate
```
7. Test installation is working
```
$ tensorflow.active
(targetDirectory)$ python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
'Hello, TensorFlow!'
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a + b)
42
>>> sess.close()
>>> exit()
(targetDirectory)$ deactivate
```

## Install python libraries for tf object detector building 
```
sudo apt-get install protobuf-compiler python-tk

tensorflow.active

pip install Cython
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
```

## COCO API installation
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools (3)/research/
```

## Install protobuf 3 on Ubuntu
https://gist.github.com/nhancv/c2fbe739f27e276ba1a36f33890e8b2a
```
# Make sure you grab the latest version
curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip

# Unzip
unzip protoc-3.2.0-linux-x86_64.zip -d protoc3

# Move protoc to /usr/local/bin/
sudo mv protoc3/bin/* /usr/local/bin/

# Move protoc3/include to /usr/local/include/
sudo mv protoc3/include/* /usr/local/include/

# Optional: change owner
sudo chwon [user] /usr/local/bin/protoc
sudo chwon -R [user] /usr/local/include/google
```

## Protobuf compilation
```
# From (3)/research
protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python setup.py build
python setup.py install
python object_detection/builders/model_builder_test.py
jupyter notebook object_detection/object_detection_tutorial.ipynb
```

## Configure your own object detector
  1. Gather data
      - Gather Pictures: can use your phone or goolge search to collect at least 200 pics (jpg format) overall, 40 pictures for each class. The picture size should be less than 200KB each and resolution <= 720x1280
      - Split 20% pics for testing (put them to the (3)/research/object_detection/images/test) and the others for training (put them to the (3)/research/object_detection/images/test).
      - Label Pictures: using https://github.com/nhancv/labelImg. Once image is labeled and saved, one .xml file will be generated for each in the /test and /train directories. This will take a while! =]]

  2. Create label mapping: Open/Create and edit (3)/research/object_detection/training/labelmap.pbtxt. This file define all classes for classification with name must be matched with picture label.
      ```
      item {
        id: 1
        name: 'nine'
      }
      ```
  3. Update Tensorflow Record generating config: Open and update label map in (3)/research/object_detection/generate_tfrecord.py. From line `31`, you need update class mapping follow `labelmap.pbtxt` above.

  4. Generate Training Data: 
      - Fist gen train_labels.csv and test_labels.csv files in (3)research/object_detection/images folder.
      - Next gen Tensorflow Record for training
      ```
      # From research/object_detection
      # Gen csv file
      python xml_to_csv.py
      # Gen tfrecord
      python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
      python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
      ```
  
  5. Update models configuration: Copy config file from `(3)/research/object_detection/samples/configs/faster_rcnn_inception_v2_pets.config` to `(3)/research/object_detection/training/faster_rcnn_inception_v2_pets.config`
      - Update config path:
        ```
        Line 9. Change num_classes to the number of different objects you want the classifier to detect, which be total id in labelmap.pbtxt. 
            num_classes: 6

        Line 110. Change fine_tune_checkpoint to:
            fine_tune_checkpoint : "(3)/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

        Lines 126 and 128. In the train_input_reader section, change input_path and label_map_path to:
            input_path : "(3)/research/object_detection/train.record"
            label_map_path: "(3)/research/object_detection/training/labelmap.pbtxt"

        Line 132. Change num_examples to the number of images you have in the /images/test directory.

        Lines 140 and 142. In the eval_input_reader section, change input_path and label_map_path to:
            input_path : "(3)/research/object_detection/test.record"
            label_map_path: "(3)/research/object_detection/training/labelmap.pbtxt"

        Save the file after the changes have been made. 
        ```

## Run the training

```
# From (3)/research/object_detection
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

## Test your trained model:
> After training, frozen inference graph will be created in `(3)/research/object_detection/inference_graph` folder. You can test it by running the object_detection_image.py (or video or webcam) script.

* Make sure you have been completed follow steps:
    - Install tf
    - Install python libraries for tf object detector building
    - COCO API installation
    - Protobuf compilation 
    - Trained graph `frozen_inference_graph.pb` file
    - Label map `labelmap.pbtxt` file



### Refs: 
1. https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
2. https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
