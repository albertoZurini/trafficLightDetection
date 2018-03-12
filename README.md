# What is this?

Traffic light detection is a program which is capable of detecting the colour of a DIY traffic light model

# How does it work?

It is composed of a deep neural network, which will output a categorical output in the following form:
```
[0, 1, 0, 0]
```

# How to read the output

A red traffic light detection will look like something like this:
```
[1, 0, 0, 0]
```
A yellow:
```
[0, 1, 0, 0]
```
A green:
```
[0, 0, 1, 0]
```
An off traffic light:
```
[0, 0, 0, 1]
```

# Dependencies

- numpy
- tensorflow
- keras
- cv2
- sklearn
- jupyter

# How to train

- Collect a traffic light dataset (I've used SSD Tensorflow API to extract the traffic light from the background)
- Run dataset/Images_fixer.ipynb to make all the images the same shape
- Run Train.ipynb and export the model

# How to predict

- Place Train.ipynb outputs ('weights.h5', 'model.json') in the same directory of Predict.py
- Import Predict.py
- Run getStatsFromImage(img), where img is the numpy array of the image (cv2.imread)