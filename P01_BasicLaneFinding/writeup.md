# **Finding Lane Lines on the Road** 

## Writeup

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---
### Pipeline
See [P01.py](P01.py).<BR>
Dependencies: python3, open-cv, numpy, matplotlib, moviepy.<BR>
To generate images and videoos into [test_images_output](test_images_output) and [test_videos_output](test_videos_output) folders type in the command line:<BR><BR>
    ```
    python3 P01.py
    ```



[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Description of the pipeline
I modified several parts of the script. Tha main function to check as starting point is process_image(). It both works for videos and stills. The main steps:

1. Multi-Channel Canny: After trying your suggested solution of grayscale conversion, blur and canny, I had difficulties with challenge.mp4, and I was not able to came up with a good enough tuning.

    1. So I separated the images to RGB channels.
    1. Applied Gaussean Blur to each channel individually.
    1. Applied Canny edge detection to each channel individually.
    1. Then combined the three resulting images by taking the max values for every pixel. This resulted a more usable edge-image. (Also tried HSL but found it mediocre.)

1. I added a function to generate mask vertices, then generated a mask image and applied the mask to the output of Canny.

1. Calculated Hough-lines

1. Calculated the left and right main lines by averaging line center coordinates and slope values. For this I separated the lines to left and right sides depending on their slope values, and also filtered out lines that are close to horizontal or vertical.

1. Historical Smoothing: the resulting lines were jumping quite nervously in the videos, so I created a History class that keeps track of line stats for the 8 most recent frames, and returns a rolling average for their center and slope values.

1. Lastly I extended the smoothed lines up to the preset upper limit and down to the bottom of the image.


### 2. Potential shortcomings
This algorithm is not robust at all.
* It can work almost exclusivel on highways, where the the two sides of the lanes are drawn and clearly visible and where there are no junctions.
* It cannot handle steep turns.
* It might be easily disturbed by roads that has lines or stripes on them.
* It cannot handle over- and underexposed images and videos.
* It does not adapt to changes of the height of the horizon. The recognition quality might be affected by braking, strong acceleration and hilly terrain.
* It cannot combine information from past images. 
* etc, etc, etc.


### 3. Possible improvements to the pipeline
* Auto-exposure is probably the easiest to add.
* Instead of the two straight lines we could use a few segmenst to follow the curvature of the road.
* In the algorithm of finding the main lines there is plenty of room for improvement here. Averaging the angles instead of the m values would be a tiny bit more precise. Then filtering the line segments could be improved a lot. Also selecting which line segments belong to which main line could take into account also their positions, etc.

