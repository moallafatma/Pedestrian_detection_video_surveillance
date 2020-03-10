# Pedestrian_detection_video_surveillance-
For the second assignment of Visual Computing course at CentraleSupélec 2020

### Input:

The inputs are frames from a video-surveillance and the ground-truth bounding for the pedestrians inside the frames. 

### Outputs: 

By running the main file ("fatma_moalla.py"), the algorithm will detect the pedestrian on image "300.jpg" and save the resulted bounding_boxes as an example, then it will detect the bounding boxes on all frames using both pretrained detectors (HoG+SVM and Cascade)
\
The algorithm will then saves the scores (IoU) and the bounding boxes in the pickeled file "result.p"

### How to run:

````
python fatma_moalla.py 
````
