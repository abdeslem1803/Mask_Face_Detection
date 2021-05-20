#!/usr/bin/env python
# coding: utf-8

# In[79]:


import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# # Hyperparametres d'entrainement
# 

# In[6]:


INIT_LR = 1e-4
epochs = 20
batch_Size = 32


# In[46]:


dataset_Foldre = "Mask_Dataset"
dataset_Classes = ["with_mask","without_mask"]
data = []
labels = []
for folder in dataset_Classes:
    path = os.path.join(dataset_Foldre,folder)
    category_num = dataset_Classes.index(folder)
    for image_name in os.listdir(path):
        image_path = os.path.join(path,image_name)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category_num)        

data = np.array(data, dtype="float32")
labels = np.array(labels)
        
    


# In[47]:


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# In[48]:


(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)


# In[49]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=20,
zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")


# In[55]:


baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))


# In[58]:


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


# In[61]:


model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False


# In[65]:


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
model.summary()


# In[66]:


new_model = model.fit(aug.flow(trainX, trainY, batch_size=batch_Size), steps_per_epoch=len(trainX) // batch_Size,
                      validation_data=(testX, testY),validation_steps=len(testX) // batch_Size, epochs=EPOCHS)


# In[85]:


model.save("new_mobilenet.model")


# In[68]:


predict_model = model.predict(testX, batch_size=batch_Size)


# In[73]:


prediction = np.argmax(predict_model,axis=1)


# In[77]:


print(type(prediction))


# In[78]:


print(classification_report(testY.argmax(axis=1), prediction))


# In[83]:


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), new_model.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), new_model.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), new_model.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), new_model.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")


# In[ ]:




