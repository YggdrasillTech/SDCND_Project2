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
[image10]: ./examples/web_images.png "Web Images"
[image11]: ./examples/softmax_probabilities.png "Softmax Probabilities"
[image12]: ./examples/feature_maps.png "Feature Maps"

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
| Max pooling	      	  | 2x2 stride,  outputs 5x5x32 				          |
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

The architecture chosen was the LeNet, because is a simple architecture that works pretty well. Even though some of the problems were that it underfits easily, so in under to get LeNet perform better some of the changes are a higher depth in the convolution filters, more hidden units in the fully connected layers, the use of dropout and L2 Regularization.

The parameters that were tuned the most are Learning Rate, Dropout and L2 Regularization. L2 Regularization was kind of funny, since it takes into account the weights magnitude a higher value would probably give to much importance to low values and underfit the model. A lower value would not harm the performance and still get the effect of L2 Regularization.

---
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image10]

I decided to use these images because of the variety in their shape. All images are different in shape and in the case of the speed limit also give me the opportunity of check against other similar speed limit signs.

Images were resize at 32x32 pixel, this cause some artifacts in the images due to this process. The most clear example is image 4 in which it is possible to see noise specially in the part of the sky. This noise affects all images (even if it is not visible) and could create a potencial misclasification.

Also the Stop sign at least the letters have also noise, so is expected that all signs that contain simbols and/or letters are hard to classifie due to artifacts in compression and resize.

The edges and other geometric features seems to be retain pretty well. Even though images are pixelated edges are still pretty sharp and detectable, So I expect than simple shapes are going to be easilly clasified.

Since my architecture doesn't take into account color, only changes in brightness could be difficult to manage (to low light or areas with different level of ligth). In web images this type of features are not present but is worth mention this posibility.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                            |     Prediction	        			            		|
|:-------------------------------------:|:---------------------------------------------:|
| No Vehicles        		                | No Vehicles    						                		|
| Right-of-way at the next intersection | Right-of-way at the next intersection    			|
| Priority Road			        		        | Priority Road				                     			|
| 70 km/h	        		                  | 20 km/h	  					                  				|
| Stop		    	                        | Stop                            							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. It got wrong guessing the speed limit, this could be due to the similarities between all speed limits signs and some inclination towards a specific one.

Comparing Test Accuracy (**0.906**) against the web images Accuracy (**0.800**) is clearly a difference of **~0.1**. This is not so high if it's take into account that only 5 images were evaluated for the web test. In k_prob later on, is posible to see that for image 4 which is the one misclassified, all the probabilities were around speed signs (image 4 is 70km/h sign). So this misclasification could posible be part of the artifacts discused in the point above. A later experiment with some noise reducing technique could reveal if this is due to noise.

Taking into account that image 4 correct class is still among the Top-5 probabilities is still consistent with Validation and Tes Accuracy in the traning phase.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here is the softmax probabilities for all 5 images:

![alt text][image11]

---
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

For feature visualization I tried to replicate the work made by NVIDIA in its [white paper](https://arxiv.org/pdf/1704.07911.pdf). The process is pretty simple, I take each layer of feature maps and average them into a single feature map. Then the average map is scale to the original image size. The average and scaling is repeated over all layers. At the end a element wise multiplication between all layers is perform to obtain a visualization of which part of the image contribute more to the network guess. Here my visualization for all 5 web images:

![alt text][image12]
