# Pedestrian_detection_video_surveillance-
For the second assignment of Visual Computing course at CentraleSupélec 2020

### Input:

The inputs are frames from a video-surveillance and the ground-truth bounding for the pedestrians inside the frames. 

### Outputs: 

By running the main file ("fatma_moalla.py"), the algorithm will:

1. Detect the pedestrian on image "300.jpg" and save the resulted bounding_boxes as an example and using both pretrained detectors (HoG+SVM and Cascade)
2. Detect the bounding boxes on all frames using both pretrained detectors.
3. Save the scores (IoU) and the bounding boxes "[frame, bb_id,x,y,dx,dy]" in the pickeled file "result.p"

### How to run:

````
python fatma_moalla.py 
````
