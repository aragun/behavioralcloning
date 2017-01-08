import cv2
import json
import pandas
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import imread
from scipy.misc import imresize
get_ipython().magic('matplotlib inline')

# Loading the dataset using pandas

driving_log = pandas.read_csv('data/driving_log.csv')
print(driving_log.head())

plt.hist(driving_log['steering'], bins=30, rwidth=2/3, color='black', range=(-0.4, 0.4))
plt.title('Steering angle distribution')
plt.show()

print('Loading center, left and right images...')
X_train = np.asarray([imread('data/'+file_name.strip(), mode='RGB') for file_name in driving_log['center']])
X_left = np.asarray([imread('data/'+file_name.strip(), mode='RGB') for file_name in driving_log['left']])
X_right = np.asarray([imread('data/'+file_name.strip(), mode='RGB') for file_name in driving_log['right']])

y_train = np.asarray(driving_log['steering'])

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0)

assert(len(X_train) == len(y_train))
print('X_train.shape:', X_train.shape)
print('X_val.shape:', X_val.shape)

# The following functions form the preprocessing pipeline.
# random_brightness function takes as input an image, converts it to
# HSV mode, changes the V channel randomly and then converts the image
# back to RGB mode.
def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

# random_flip flips the image and the corresponding steering angle randomly.
def random_flip(image,steering):
    coin=np.random.randint(0,2)
    if coin==0:
        image,steering=cv2.flip(image,1),-steering
    return image,steering

# random_shear. This is done using two triangles, representd by tr1 and tr2.
# tr2 is obtained using a random_point created using the shear_range parameter.
# Using the getAffineTransform, a transformation matrix is obtained and then
# applied to the entire image using the warpAffine function.
# For the steering angle modification, the small angle approximation of 
# tan(x) ≈ x is used.
def random_shear(image,steering,shear_range):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    random_point = [cols/2+dx,rows/2]
    tr1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    tr2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/rows   
    M = cv2.getAffineTransform(tr1,tr2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering  
    return image,steering

# A region of interest is chosen, in this case ignoring the top 50 (representin
# the major portion of the horizon) and bottom 15 pixels (representing the 
# bonnet) are removed.
def region_of_interest(image):
    return image[50:-15,:]

# The image is normalized to have values between -1 and 1.
def normalize(image):
    return image/127.5 - 1.0

# A random image is chosen from the left, right and center images.
# An approximate steeing angle of 0.2 is added and subtracted from the 
# left and right images respectively. This is because of the intuition
# that if you the car's center camera was actually seeing what the left 
# camera image is showing, the car would have to steer 0.2 to the right 
# and similarly for the right camera image.
def read_next_image(m,lcr,X_train,y_train):
    steering = y_train[m]
    if lcr == 0:
        image = X_left[m]
        steering += 0.2
    elif lcr == 1:
        image = X_train[m]
    elif lcr == 2:
        image = X_right[m]
        steering -= 0.2
    else:
        print ('Invalid lcr value :',lcr )
    return image,steering

# This calls all the preprocessing functions above.
def preprocessing_pipeline(image, steering):
    image = random_brightness(image)
    image,steering = random_flip(image,steering)
    image,steering = random_shear(image,steering,shear_range=80)
    image = normalize(image)
    image = region_of_interest(image)
    image = imresize(image, (66,200,3))
    return image, steering

def generate_training_example(X_train,X_left,X_right,y_train):
    m = np.random.randint(0,len(X_train))
    lcr = np.random.randint(0,3)
    image,steering = read_next_image(m,lcr,X_train,y_train)
    return preprocessing_pipeline(image,steering)

image, steering = generate_training_example(X_train,X_left,X_right,y_train)

# This is the keras generator used to generate training samples on the fly
# using the preprocessing functions defined above. 
def generate_train_batch(X_train,X_left,X_right,y_train,batch_size = 32):    
    batch_images = np.zeros((batch_size, 66, 200, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            x,y = generate_training_example(X_train,X_left,X_right,y_train)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering
        
batch_size=200
train_generator = generate_train_batch(X_train,X_left,X_right,y_train,batch_size)

# Normalizing and resizing validation set.
X_val = np.asarray([imresize(normalize(image), (66,200,3)) for image in X_val])
print('Resized X_val.shape:', X_val.shape)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2

# This section creates and runs the keras model. l2 regularizers have been
# used to prevent overfitting.
model = Sequential()
model.add(Conv2D(24, 5, 5, subsample=(2,2), input_shape=(66, 200, 3), activation='relu', W_regularizer=l2(0.01)))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu', W_regularizer=l2(0.01)))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu', W_regularizer=l2(0.01)))
model.add(Conv2D(64, 3, 3, subsample=(1,1), activation='relu', W_regularizer=l2(0.01)))
model.add(Conv2D(64, 3, 3, subsample=(1,1), activation='relu', W_regularizer=l2(0.01)))
model.add(Flatten())
model.add(Dense(1164, activation='relu', W_regularizer=l2(0.01)))
model.add(Dense(100, activation='relu', W_regularizer=l2(0.01)))
model.add(Dense(50, activation='relu', W_regularizer=l2(0.01)))
model.add(Dense(10, W_regularizer=l2(0.01)))
model.add(Dense(1, W_regularizer=l2(0.01)))

model.summary()

adam = Adam(lr=1e-4)

from keras.models import model_from_json
model_json = 'model.json'
model_weights = 'model.h5'

reuse=True
if os.path.isfile(model_json) and reuse:
    try:
        with open(model_json) as jfile:
            model = model_from_json(json.load(jfile))
            model.load_weights(model_weights)    
        print('loading trained model ...')
    except Exception as e:
        print('Unable to load model:', e)
        raise 

model.compile(optimizer=adam, loss='mse', metrics=['mean_squared_error'])

history = model.fit_generator(train_generator,
                    samples_per_epoch=40000,
                    nb_epoch=5,
                    verbose=1, validation_data=(X_val, y_val))

try:
    os.remove(model_json)
    os.remove(model_weights)
except OSError:
    pass 

# Saving the model architecture and weights locally.
model_json_string = model.to_json()
with open(model_json, 'w') as outfile:
    json.dump(model_json_string, outfile)
model.save_weights(model_weights)

# This section just tests the model on a random image to see
# how well the model performs.
m = np.random.randint(0,len(X_train))
ximg = imresize(normalize(X_train[m]), (66,200,3))
plt.imshow(ximg)
ximg = ximg[None, :, :, :]
print(ximg.shape)
model.predict(ximg, batch_size=1, verbose=1)
angle = float(model.predict(ximg, batch_size=1, verbose=1))
print('Prediction:{}'.format(angle))
print('Actual steering angle:{}'.format(y_train[m]))