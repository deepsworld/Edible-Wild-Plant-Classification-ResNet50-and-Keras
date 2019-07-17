# Edible-Wild-Plant-Classification-ResNet50-and-Keras

 A deep learning classifier that uses pre-trained CNN model to classify 54
categories of wild edible plants into their respective classes. 

A Resnet50 model pre-trained on the ImageNet dataset was used as the base model to extract the
features from the plant images. Only the top few layers of the Resnet50 model were trained using
the edible plants dataset. The model was further finetuned so that it adapts to the new task at
hand and to improve the accuracy. We added three other layers on top of the base model to
prevent issues that might arise due to the small dataset and reduce the training time. In the
experiment, we compared the validation accuracy of the model before and after finetuning the
model. And we used confusion matrix and classification report to evaluate the effectiveness of
the pre-trained Resnet50 classifier. The results of our experiment show that the pre-trained neural
network was effectively able to classify wild edible plants. We were able to achieve even better
results after fine tuning the model. 


The data set, data augmentation and data preprocessing
We collected the images of 54 different wild edible plants from Flickr using their API available
for non-commercial outside developers. The training dataset consist of 7535 images and
validation dataset contains 734 images of various size. The distribution of the training and testing
data is shown in figure 1. The original data is very small considering the large number of classes.
It is very easy to overfit the neural network with our dataset. We use a widely used and a simple
technique called data augmentation to prevent overfitting [1]. We applied transformation
methods such as, horizontal flip, rotation, shift, shear, scaling and zoom to increase the size of
our dataset. The example of image augmentation is shown in the figure 2.
Every single original image is transformed into 9 different images while retaining the vital
features. All the images were transformed from their original size to 224x224 as Resnet50
requires any input image to be of 224x224. The images larger than the required size were resized
whereas smaller sized images were padded using the nearby pixel values. It is also advantageous
for the gradient descent in the training process to normalize the raw pixels of the images.
Therefore, we normalized the pixel data of each image between 0-255 to between 0-1. 
