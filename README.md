# **Traffic Sign Recognition**

This **README** file serve as write-up for the Udacity SDCND Project2

---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training_distribution.png "Training Distribution Visualization"
[image2]: ./examples/validation_distribution.png "Validation Distribution Visualization"
[image3]: ./examples/test_distribution.png "Validation Distribution Visualization"
[image4]: ./examples/gray_scale.png "Gray Scale Examples"
[image5]: ./examples/random_rotation.png "Random Rotation Examples"
[image6]: ./examples/random_translation.png "Random Translation Examples"
[image7]: ./examples/normalized.png "Normalized Images Examples"
[image8]: ./examples/loss_history.png "Loss History Graph"
[image9]: ./examples/accuracy_history.png "Accuracy History Graph"

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

For the basic exploratory information I used Numpy. With the us of np.shape I obtain the dimensions of each of the datasets with the following results:

* The size of training set is **34799**.
* The size of the validation set is **4410**.
* The size of test set is **12630**.
* The shape of a traffic sign image is **(32, 32, 3)**.
* The number of unique classes/labels in the data set is **43**.

#### 2. Include an exploratory visualization of the dataset.

I obtain the distribution of each of the classes for the three data sets to see how each class is represented.  Beginning with the Training Dataset distribution.

![alt text][image1]

Now the Validation Dataset.

![alt text][image2]

And lastly Test Dataset.

![alt text][image3]

I can see that the three distributions are pretty similar even though they have some minor difference, this could ensure that a similar performance between the three Datasets.

---
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The first step in the pre processing pipeline was to convert images into gray scale images. This help to lower the number of channels the net has to deal.

![alt text][image4]

Within my pipeline I also use **data augmentation** within the training dataset. For data augmentation I use only random rotation and random translation. Here you can see some examples of random rotation:

![alt text][image5]

And random Translation examples:

![alt text][image6]

Finally a normalization step was perform to restrict the values of images between -1.0 and 1.0, there are no big differences between gray scale and normalized images due to matplot lib plotting.

![alt text][image7]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		    |     Description	        					            |
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x1 gray scale image   							      |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					        |												                        |
| Dropout               | Training Dropout Probability of 0.3           |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x16 				        |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					        |												                        |
| Dropout               | Training Dropout Probability of 0.3           |
| Max pooling	      	  | 2x2 stride,  outputs 5x15x32 				          |
| Flatten       	      | outputs 800                 									|
| Fully connected		    | inputs 800, outputs 200      									|
| Fully connected		    | inputs 200, outputs 100      									|
| Fully connected		    | inputs 100, outputs 43      									|
| Softmax				        | Performs Softmax Probabilities                |

This is my final architecture which is a modified version of LeNet Architecture.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model the following hyperparameters were used:
* Epoch: **200** (even though a conditional was use to stop training at **0.93** validation accuracy).
* Learning Rate: **0.00005**
* Batch Size: **32**
* L2 Regularization: **0.0001**

For the optimizer I used Adam Optimizer since it consistently show good results, for loss the use of cross entropy and L2 regularization over all weights in layers were used. L2 Regularization helps to prevent overfitting at the same time it keeps weight values pretty low.

At the end the model just took 43 epochs to reached **0.930** of validation accuracy. We can see how the loss and accuracy behaved in the model with the next graphs:

Loss History:

![alt text][image8]

Accuracy History:

![alt text][image9]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **0.968**
* validation set accuracy of **0.930**
* test set accuracy of **0.906**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

---
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:



The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

---
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
