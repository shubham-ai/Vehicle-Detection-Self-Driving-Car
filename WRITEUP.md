
##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)



### The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
    * Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/randome_car_img.png
[image7]: ./output_images/hist_img.png
[image8]: ./output_images/bin_spatial_img.png
[image2]: ./output_images/hog_img.png
[image3]: ./output_images/sliding_window.png
[image4]: ./output_images/search_windows_img.png
[image5]: ./output_images/pipe_heatmap_img.png
[image6]: ./output_images/final_img.png
[image9]: ./output_images/pipe_heatmap_img.png

pipe_draw_rec_img.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

#### Below are the steps described individually that are implement in the project


1) HOG feature
2) Extract Feature(SVC , CLF Results)
3) Sliding Windows
4) Pipeline Explanation



###Histogram of Oriented Gradients (HOG)

####1.1) HOG feature  is called Histogram of Oriented Gradient
* It has very import task in prediction as it helps to get result
* It mainly consist of gradient , magnitued and direction
* Grouping these individual values into small group of cells
* And classifing from almost 9 orients bins to get the result
* and then combining again the each pixel


I started by reading cutout images of cars

such as

![alt text][image1]

then I Compute the histogram of the RGB channels separately using function color_hist which baiscally compute the histogram of RGB channel saperately , generate bin centers and then concate them into single channel later return Individuals hist, bin and feature veactor

![alt text][image7]


I then working on bin_spatial() function which i basically think resize the feature image using ravel and cv2.resize but we do it by imposing on singular color section of RGB and then we use hstack to stack them into one unit of list

![bin_spatial][image8]

I then extracted all the images and read there data using function data_look().

After that i worked on get_hog_features() function and used hog by using  `import skimage.feature import hog`


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][image2]

furthen i used extract_features function
and implemented all above togather

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:



####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters but the best result i got via
```python
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
0.7 Seconds to train SVC...
Test Accuracy of SVC =  1.0
179.53 Seconds to train CLF...
Test Accuracy of SVC =  1.0
My CLF predicts:  [ 0.  0.  1.  1.  1.  1.  0.  0.  0.  0.]
For these 10 labels:  [ 0.  0.  1.  1.  1.  1.  0.  0.  0.  0.]
0.078 Seconds to predict 10 labels with CLF
```


below are by resultant predictoin

```python

Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
0.7 Seconds to train SVC...
Test Accuracy of SVC =  1.0
179.53 Seconds to train CLF...
Test Accuracy of SVC =  1.0
My CLF predicts:  [ 0.  0.  1.  1.  1.  1.  0.  0.  0.  0.]
For these 10 labels:  [ 0.  0.  1.  1.  1.  1.  0.  0.  0.  0.]
0.078 Seconds to predict 10 labels with CLF

```
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

i trained the mode via SVM and CLF and found CLF is way better than SVM  as the prediction result Drastically  got accruate by using CLF or SVM  
though i used LinearSVC

```python

svc = LinearSVC(loss='hinge')
svc.fit(X_train, y_train)

round(svc.score(X_test, y_test), 4))


```


ClF code below

```python
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = SVC()
clf = GridSearchCV(svr, parameters)

```

![hog][image2]

###Sliding Window Search

####1. I Implemented a sliding-window technique and use your trained classifier to search for vehicles in images. The basic sliding windows function i use is slide_window() in which I manually  gave x and y cooridnated in order to get the sliding windows result basically learned how it workes from  scartch

the i got into pro mode :P and end up implementing search_windows function which further Includes  single_img_features in which I used CLF as one of the key parameter and predicted the cars



![alt text][image3]

![alt text][image4]



####2.my piple line i.e process_pipe() include image as input then has combination of find car
function utility at variout y-coordinates in order to perdict the accuratie result as distance matter for the shape and size of car and most importantly i read image via mping library which is RGB format because of which i have to remove

```python
img = img.astype(np.float32)/255
```

for accurate results, I guess we need some clearification from udacity on this note as color space does matter a lot and it took me lot of time to figure it out and agian CLF is Awesome



![alt text][image6]
---

### Video Implementation


Here's a [link to my video result](./project_video.mp4)




### Here are six frames and their corresponding heatmaps:

I have use three function add_heat()  ,apply_threshold() and draw_labeled_bboxes

in which multiple detections are predicted and false head is removed  ,

all there are combined in function process_pipe  ,

so we will build a heat-map from  detections in order to combine overlapping detections and remove false positives.

then threshold is used in order to remove the area having false positive as it's not much dense and is turn out via conditonon

```python
heatmap[heatmap <= threshold] = 0
```


later draw_labeled_bboxes function is used and result is achieved


![alt text][image5]

_i have used Deque method to collect record of past frame and sum them up to get better results , thanks to my udacity  , i hope you have like my video_

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![heat][image6]




---

###Discussion

#### 1.  At some point and in very low leve it looks like it can make boxes on wrong spot

#### 2. I also experice problem when i used different parameters for CLF and SVC form  paremeters  of find_car function

#### 3 I have to elimate _*img = img.astype(np.float32)/255*_  from find car fucntion in order to get accurate result


belwo are the video that are created after learning

##### test video
<video width="400" controls>
  <source src="test_video_out.mp4.mp4" type="video/mp4">
</video>

##### project video
<video width="400" controls>
  <source src="project_video_out.mp4.mp4" type="video/mp4">




```python

```
