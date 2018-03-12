
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import os
import cv2


# In[3]:


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("weights.h5")
print("Loaded model from disk")

# Compile the model
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[4]:


def getStasFromImage(img):
  # Adjuse channel position basing on the backend
  # Every CNN layer has to know where the number of channels is
  # It can be channel_first or channel_last
  width, height = 72, 98
  img = cv2.resize(img, (width, height))
  if K.image_data_format() == 'channels_first':
    img = img.reshape(1, 3, height, width)
  else:
    img = img.reshape(1, height, width, 3)
  
  #img = img.astype('float32')
  #img /= 255

  return loaded_model.predict(img)


# In[14]:


# Converts the predictions array into a number
def catToNum(arr):
  high = 0
  highIndex = 0 # The number
  i = 0
  for num in arr[0]:
      if num > high:
          high = num
          highIndex = i
      i += 1
  return highIndex

