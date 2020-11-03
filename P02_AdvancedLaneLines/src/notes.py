# TODO use P01's Multi-Channel Canny
# TODO idea: channel by channel equalizeHist, up to a limit
# TODO idea: use convolutions shaped: / \ and | (Sobel)
    # https://classroom.udacity.com/nanodegrees/nd013-ent/parts/f114da49-70ce-4ebe-b88c-e0ab576aed25/modules/781fbec1-2de6-4c9f-9407-08c010b5bc7e/lessons/4883b3e6-1679-48d3-8fa3-b5893659d657/concepts/e6115672-155d-4c10-b640-fe20a4f4b0a6
    # https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
# TODO idea: use the gradient of previously found line to choose which convolution to use
# TODO idea: for horizon detection use convolutions shaped: - (Sobel)
# TODO idea: use memory for masking: mask out parts that are far from previous frame's good quality (!) line
# TODO idea: use the time spent since last good quality line finding to increase the width of the mask


