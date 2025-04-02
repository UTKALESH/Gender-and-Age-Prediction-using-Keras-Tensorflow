#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input


# In[5]:


file_path = r"D:\Downloads\archive (4)\UTKFace"


df = pd.DataFrame({'image': os.listdir(file_path)})


df['image'] = df['image'].apply(lambda x: os.path.join(file_path, x))

print(df.head())


# In[6]:


# labels - age, gender, ethnicity
image_paths = []
age_labels = []
gender_labels = []

for filename in tqdm(os.listdir(file_path)):
    image_path = os.path.join(file_path, filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)


# In[7]:


# convert to dataframe
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels
df.head()


# In[8]:


# map labels for gender
gender_dict = {0:'Male', 1:'Female'}


# In[9]:


from PIL import Image
img = Image.open(df['image'][0])
plt.axis('off')
plt.imshow(img);


# In[10]:


sns.distplot(df['age'])


# In[11]:


sns.countplot(df['gender'])


# In[12]:


# to display grid of images
plt.figure(figsize=(20, 20))
files = df.iloc[0:25]

for index, file, age, gender in files.itertuples():
    plt.subplot(5, 5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(f"Age: {age} Gender: {gender_dict[gender]}")
    plt.axis('off')


# In[13]:


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode="grayscale")
        img = img.resize((128, 128), Image.LANCZOS)
        img = np.array(img)
        features.append(img)
    return np.array(features)


# In[14]:


X = extract_features(df['image'])


# In[15]:


X.shape


# In[16]:


X = X/255.0


# In[17]:


y_gender = np.array(df['gender'])
y_age = np.array(df['age'])


# In[18]:


input_shape = (128, 128, 1)


# In[19]:


inputs = Input((input_shape))
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2)) (conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu') (maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu') (maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2)) (conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2)) (conv_4)

flatten = Flatten() (maxp_4)


dense_1 = Dense(256, activation='relu') (flatten)
dense_2 = Dense(256, activation='relu') (flatten)

dropout_1 = Dropout(0.3) (dense_1)
dropout_2 = Dropout(0.3) (dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
output_2 = Dense(1, activation='relu', name='age_out') (dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])


# In[23]:


model.compile(
    loss=['binary_crossentropy', 'mae'], 
    optimizer='adam',
    metrics=[['accuracy'], ['mae']]  
)


history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=30, validation_split=0.2)


# In[24]:


train_acc = history.history.get('gender_out_accuracy', [])
val_acc = history.history.get('val_gender_out_accuracy', [])


train_loss = history.history.get('gender_out_loss', [])
val_loss = history.history.get('val_gender_out_loss', [])


epochs_range = range(1, len(train_acc) + 1)


plt.figure(figsize=(10, 4))
plt.plot(epochs_range, train_acc, linestyle='-', color='blue', marker='o', label='Train Accuracy')
plt.plot(epochs_range, val_acc, linestyle='--', color='red', marker='s', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(epochs_range, train_loss, linestyle='-', color='blue', marker='o', label='Train Loss')
plt.plot(epochs_range, val_loss, linestyle='--', color='red', marker='s', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


train_loss = history.history['age_out_loss']
validation_loss = history.history['val_age_out_loss']
epochs_range = range(len(train_loss))


plt.figure(figsize=(8, 5))

# Plot training and validation loss
plt.plot(epochs_range, train_loss, linestyle='--', color='blue', label='Train Loss')
plt.plot(epochs_range, validation_loss, linestyle='-', color='red', label='Validation Loss')

# Enhance visualization
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.title('Training vs Validation Loss for Age Prediction')
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.5)

# Display the plot
plt.show()


# In[26]:


# Select an image index for testing
sample_idx = 100

# Display the actual gender and age
true_gender = gender_dict[y_gender[sample_idx]]
true_age = y_age[sample_idx]
print(f"Actual → Gender: {true_gender} | Age: {true_age}")

# Reshape the image and make a prediction
sample_image = X[sample_idx].reshape(1, 128, 128, 1)
prediction = model.predict(sample_image)

# Process model predictions
predicted_gender = gender_dict[round(prediction[0][0][0])]
predicted_age = round(prediction[1][0][0])
print(f"Predicted → Gender: {predicted_gender} | Age: {predicted_age}")

# Display the image
plt.figure(figsize=(4, 4))
plt.imshow(X[sample_idx].reshape(128, 128), cmap='bone')  # Changed colormap
plt.axis('off')
plt.title(f"Predicted: {predicted_gender}, Age: {predicted_age}")
plt.show()


# In[27]:


# Select an image index for testing
sample_idx = 1000

# Display the actual gender and age
true_gender = gender_dict[y_gender[sample_idx]]
true_age = y_age[sample_idx]
print(f"Actual → Gender: {true_gender} | Age: {true_age}")

# Reshape the image and make a prediction
sample_image = X[sample_idx].reshape(1, 128, 128, 1)
prediction = model.predict(sample_image)

# Process model predictions
predicted_gender = gender_dict[round(prediction[0][0][0])]
predicted_age = round(prediction[1][0][0])
print(f"Predicted → Gender: {predicted_gender} | Age: {predicted_age}")

# Display the image
plt.figure(figsize=(4, 4))
plt.imshow(X[sample_idx].reshape(128, 128), cmap='bone')  # Changed colormap
plt.axis('off')
plt.title(f"Predicted: {predicted_gender}, Age: {predicted_age}")
plt.show()


# In[28]:


# Select an image index for testing
sample_idx = 9000

# Display the actual gender and age
true_gender = gender_dict[y_gender[sample_idx]]
true_age = y_age[sample_idx]
print(f"Actual → Gender: {true_gender} | Age: {true_age}")

# Reshape the image and make a prediction
sample_image = X[sample_idx].reshape(1, 128, 128, 1)
prediction = model.predict(sample_image)

# Process model predictions
predicted_gender = gender_dict[round(prediction[0][0][0])]
predicted_age = round(prediction[1][0][0])
print(f"Predicted → Gender: {predicted_gender} | Age: {predicted_age}")

# Display the image
plt.figure(figsize=(4, 4))
plt.imshow(X[sample_idx].reshape(128, 128), cmap='bone')  # Changed colormap
plt.axis('off')
plt.title(f"Predicted: {predicted_gender}, Age: {predicted_age}")
plt.show()


# In[ ]:




