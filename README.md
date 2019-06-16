# Image Caption Generator

Requires latest version of python (Python 3.7.3) otherwise it wont work.

## Standard

Follows PEP 8 coding style

## Requirements

1. Tensorflow
2. Keras
3. Numpy
4. h5py
5. Pandas
6. Pillow

## Dataset

The dataset used is flickr8k. Data can be requested the data [here](https://illinois.edu/fb/sec/1713398). An email for the links
of the data to be downloaded will be mailed to your id. Extract the images in images folder and the text data in captions.

Trained model lives in directory named models.

Most folder are gitignored so manual creation should be done

## Instructions

Preprocess which add start and end token
Train the data
Test the image from test dir

```bash
python test.py image_name
```
