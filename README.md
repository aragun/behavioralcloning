# Behavioral Cloning

## Network Architecture
The deep neural network architecture from NVIDIA's End to End Learning for Self-Driving Cars paper (https://arxiv.org/pdf/1604.07316.pdf) was used. It uses 5 convolutional layers followed by 4 fully connected layers and has close to 1.5 million parameters.

## Preprocessing:
1. Images from the center, left and right cameras were used. An offset of 0.2 was added and subtracted, respectively, from the steering angles for the left and right camera images.
2. A Keras generator was used to generate training data on the fly. It did the following:
  * Change brightness of images randomly by converting them to the HSV channel and changing the V channel randomly.
  * Randomly flip images and reverse the steering angle
  * Shear the image using a random point, and then using the getAffineTransform and warpAffine opencv funtions to create a transformed image. The steering angle was also changed appropriately using the small angle approximation of tan(x) ≈ x
  * A region of interest was selected from the image (which removed some pixels from above the horizon as well some from the car bonnet.
  * The image pixel intensities were normalized to have values between -0.5 and 0.5
  * The image was resizd to (66,200,3) for input to the NVIDIA network.
  
## Training
1. The best results were obtained for 40000 samples per epoch over 5 epochs. The training data was collected by manually driving over portions of the track.
2. Relu activation units were used to introduce non-linearity in the neural network.
3. L2 regularization was used in all layers to prevent overfitting. Dropout didn't turn out to be as useful as using L2 regularization.
4. Adam optimizer with a learning rate of 0.0001 was used. 
5. A validation set was used during training. However, since mean squared error does not seem a very useful loss function for this task (since most of the steering angles are close to 0), the usefulness of the validation set is unclear.

## Results
The model is able to drive around the test track safely. Watch the video [here.](https://www.youtube.com/watch?v=S6y2VIWvR6A) 
