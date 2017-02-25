##P5 writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar_sample.png
[image2]: ./output_images/car_hog.png
[image9]: ./output_images/notcar_hog.png
[image3]: ./output_images/window_search.png
[image4]: ./output_images/window_search_example1.png
[image5]: ./output_images/window_search_example2.png
[image6]: ./output_images/window_search_example3.png
[image7]: ./output_images/single_frame_pipeline_original.png
[image8]: ./output_images/single_frame_pipeline_multiple_bboxes.png
[image10]: ./output_images/single_frame_pipeline_heatmap.png
[image11]: ./output_images/single_frame_pipeline_labeled.png
[image12]: ./output_images/single_frame_pipeline_final.png

[image15]:./output_images/cross_frame_heatmap.png
[image16]:./output_images/cross_frame_labeled.png
[image17]:./output_images/cross_frame_final.png


[image31]:./output_images/cross_frame_1.png
[image32]:./output_images/cross_frame_2.png
[image33]:./output_images/cross_frame_3.png
[image34]:./output_images/cross_frame_4.png
[image35]:./output_images/cross_frame_5.png
[image36]:./output_images/cross_frame_6.png
[image37]:./output_images/cross_frame_7.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features` of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `GRAY` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image9]


####2. Explain how you settled on your final choice of HOG parameters.

I mainly tried different colorspaces, eg. `YCrCb` with all channels, `U` in `LUV` (the one which worked for me in Project #4), but I noticed there is not much difference among them. So to reduce features I chose the simplest grayscale image. 

I did not tweak other HOG parameters that much. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

To train the classifier, first I obtained a feature vector by concatenating HOG features + color histogram features (with original RGB image) + spatial features. Then I normalized using `StandardScalar()`. The final feature vector has a length of 2580.

I used a Linear SVM as my classifier. 

For the training data, I split all vehicle + nonvehicle data randomly into 90% training and 10% test data, using sklearn's `train_test_split` method. I also run a 5-fold cross validation.

The test accuracy of the classifier is around 97%. 

Code for this step is in cell 7-9. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used 3 sizes of windows at different portions of image: 

- (80,80) and (96,96) for the lower half of the image (excluding left regions)
- (128,128) for the right-bottom quarter of the image. This is mainly to catch entering vehicles.

I used HOG sub-sampling method in Lesson 33 (code in `find_cars()` function). With a `cells_per_step=2`, each window size translate to these overlaps: 

- (80,80): 10 cells in one window -> 80% overlap
- (96,96): 12 cells in one window -> 10/12 ~ 85% overlap
- (128,128): 16 cells in one window -> 14/16 = 87.5% overlap

It looks like this with all boxes plotted: 

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some example images:

![alt text][image4]
![alt text][image5]
![alt text][image6]

I used Python's `multiprocessing` module to speed up subsampling of HOG features, using a `Pool` of 12 threads. The parallel code can be found in function `find_cars`, specifically this line: 

```
pool.map(partial(hog_parallel.get_hog_feature_parallel, hogs = [hog1], nblocks_per_window = nblocks_per_window), fs)
```

where `hog_parallel` is a separate Python module (see `hog_parallel.py` file) I imported into notebook to work around the issue that iPython notebook does not support `multiprocessing` out of box on Windows. 

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output_submission.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used a "double-heatmap" approach, applying one heatmap per frame, then applying another heatmap across frames.

First I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

You can find implementation for single frame pipeline in `process_image` function. 

Here's an example result showing each stage **per frame:** 

1. original image 

    ![alt text][image7]

2. image with multiple bounding boxes
    
	![alt text][image8]

3. image with heatmap 
    
	![alt text][image10]

4. image with `label()`
    
	![alt text][image11]

5. image with final bounding boxes. 
    
	![alt text][image12]

Here's an example result showing the pipeline **for a series of frames of video**: after recognizing vehicles in each frame, I apply another heatmap for 7 consecutive frames' bounding boxes. 

The code can be found in `video_process_image()` function.

1. 7 consecutive frames with their bounding box (after heatmap already)
    ![alt text][image31]
    ![alt text][image32]
    ![alt text][image33]
    ![alt text][image34]
    ![alt text][image35]
    ![alt text][image36]
    ![alt text][image37]
2. integrated heatmap
    
	![alt text][image15]

3. output of applying `labels()` on integrated heatmap
    
	![alt text][image16]

4. resulting bounding boxes for the last frame
    
	![alt text][image17]

In this case it seems the final integrated bounding boxes are worse (not as tight as) than the ones obtained through single-frame-only pipeline. However, running through the whole video has proved that this approach stabalizes detections significantly.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First I had performance problems: originally my pipeline takes 10 seconds to render one frame, and this means the whole project video takes 3 hours. 

I tried to speed up with Python's `multiprocessing` module, but it did not work well with Jupyter, so in the end I had to import my hog extraction function as a separate module. After that, I was able to render 1 frame per second. 

The current pipeline does not work well if there's a car in the middle of the screen, eg. a car right in front. It also does not detect very smoothly when a car just enters the scene. 

One potential improvement is to use dynamic window sizes in searching, for example, use larger windows for closer areas and smaller windows for further areas, and align it dynamically with detected lanes.  

