#!/usr/bin/env python
# coding: utf-8

# # Assignement 2
# 
# This is your second assignement in the computer vision course. This time you are supposed to implement a system that outputs bounding boxes (bb) of pedestrians.
# 
# **Bounding box definition:** the smallest axis aligned rectangle containing all the pixels of the pedestrian
# 
# !!NO DEEP LEARNING APPROACHES ARE ALLOWED. You should use classic methods to address the assignment!!
# 
# Ouput format:
#  - A list of [frame_id, bb_id, x, y, dx, dy]. See prenom_nom.py module for an example
#  
# Evaluation function:
#  - Detections are converted into binary masks and we calculate the following metrics: intersecion over union. Note that this metric is not sensitive to the number of bb-s. You can find the code below for the evaluation.
#  - The evaluation will be performed in an independent video sequence.
#  
# Scoring:
#  - Your work will be evaluated as usual (complexity of the solution, clean implementation, well documented) **PLUS** the best 5 will receive +1 for the grade of the assignement.
#  
# You have to handle in:
#  - Your code that is a single python module (possibly with requirements.txt or with a dockerfile). See the example prenom_nom.py. It has to implement the same interface, if it fails to run, your solution is considered failing.
#  - Your report. Short summary of your algorithm, motivation for the algorithm used, failing cases, code and results (~1 page).
#  
# You should send your assignment by mail to maria.vakalopoulou@centralesupelec.fr, the name of the subject of the mail should be: VIC_Assignement2_name
#  

### Libs

from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import yourname
import tqdm
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
import pickle

### inputs
data_root = './img1/'
gt_path = './gt/gt.txt'

_W = 1280
_H = 960
_N = 684 # number of frames

### Utils functions
def format_id(frame):
    """Add formatting to this id '1'--> '001'"""
    assert _N >= frame
    return '{:03d}'.format(frame)


def read_frame(root, frame):
    """Read frames and create integer frame_id-s"""
    assert _N >= frame
    return cv2.imread(os.path.join(root,format_id(frame)+'.jpg'), cv2.IMREAD_UNCHANGED)
    

def read_gt(filename):
    """Read gt and create list of bb-s"""
    """ Gt : Ground truth bounding boxes"""
    assert os.path.exists(filename)
    with open(filename, 'r') as file:
        lines = file.readlines()
    # truncate data (last columns are not needed)
    return [list(map(lambda x: int(x), line.split(',')[:6])) for line in lines]


def annotations_for_frame(solution, frame):
    assert _N >= frame
    return [bb for bb in solution if int(bb[0])==int(frame)]


def evaluate_solution(gt, solution, N):
    """Caclulate evaluation metric"""
    score = []
    #for frame in [300]:
    for frame in range(1, N):
        bbs_sol = annotations_for_frame(solution, frame)
        bbs_gt = annotations_for_frame(gt, frame)
        black_sol = np.zeros((_H, _W))
        black_gt = np.zeros((_H, _W))
        for bb in bbs_sol:
            x, y = bb[2:4]
            dx, dy = bb[4:6]
            cv2.rectangle(black_sol, (x, y), (x+dx, y+dy), (255), -1)
        for bb in bbs_gt:
            x, y = bb[2:4]
            dx, dy = bb[4:6]
            cv2.rectangle(black_gt, (x, y), (x+dx, y+dy), (255), -1)
        # intersection over union
        intersection = black_sol * black_gt
        intersection[intersection > 0.5] = 1
        union = black_sol + black_gt
        union[union > 0.5] = 1
        if not union.any():
            continue
        score.append(intersection.sum()/union.sum())    
    return np.asarray(score).mean()
    

def show_annotation(solution, frame):
    assert _N >= frame
    im = read_frame(data_root, frame)
    bbs = annotations_for_frame(solution, frame)
    for bb in bbs:
        x, y = bb[2:4]
        dx, dy = bb[4:6]
        cv2.rectangle(im, (x, y), (x+dx, y+dy), (0,255,0), 10)
    #plt.imshow(im)
    #plt.title('Annotations for frame {}.'.format(frame))
    #plt.show()
    return

 ### Solution starts here
def read_show(imagePath):
    """
    This function reads and returns the frame_n and the image 
    ---
    input : imagePath ('./folder_name/frame_n.jpg')
    ---
    output : image and frame_n
    """
    frame_n = int(imagePath.split('/')[2].split('.jpg')[0])
    print('Processing image {} ...'.format(frame_n))
    image = cv2.imread(imagePath) 
    image = imutils.resize(image, width= image.shape[1]) 
   # plt.imshow(image)
   # plt.title('Original image {}'.format(frame_n))
   # plt.show()
    return image, frame_n
    
    

def pedestrian_detect1(imagePath,nms=True):
    """
    This fuction detects pedestrians based on HOG +SVM + OpenCV pedestrian detector 
   
        ---
    input : imagePath : ('./folder_name/frame_n.jpg') + Gray : use grascale
    ---
    output : Boundingbox [x,y,dx,dy]
    
    """
    image, frame_n= read_show(imagePath)
    img = image.copy()
    #1. Get the HOG SVM detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    #2. detect people in the image using HOG
    (rect, weights) = hog.detectMultiScale(img, winStride=(4, 4),padding=(4, 4), scale=1.05)
    
    #3. apply non maximum supression or not 
    if nms:
        print('Using non-maximum supression')
        pick  = non_max_suppression(rect , probs=None, overlapThresh=0.2)
        print("[INFO] {}: {} original boxes, {} after suppression".format(frame_n, len(rect), len(pick)))
    else:
        print('NOT Using non-maximum supression')
        pick = rect.copy()
    
    
    #3. draw the original bounding boxes
    for (x,y,w,h) in pick:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, 'Pedestrian', (x + 6, y - 6), font, 1., (0, 255, 0), 3)
       # plt.imshow(img)
   # plt.title('After detection {}'.format(frame_n))
    #plt.show()
    return frame_n,pick 

def pedestrian_detect2(imagePath, nms,gray=False):
    """
    This fuction detects pedestrians based on a pretrained model that uses Haarcascade 
    ---
    input : imagePath : ('./folder_name/frame_n.jpg') + Gray : use grascale + nms: apply or not non-max-suppression
    ---
    output : Boundingbox [x,y,dx,dy]
    
    """
    image, frame_n= read_show(imagePath)
    
    img = image.copy()
    if gray :
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     #   plt.imshow(gray)
      #  plt.title('Original gray image {}'.format(frame_n))
        img = gray
     #   plt.show()
        
    #1. Import a pretrained classifier on another training set:
    #https://github.com/chandravenky/Computer-Vision---Object-Detection-in-Python/tree/master/xml%20files
    pedestrian_cascade = cv2.CascadeClassifier('./haarcascade_fullbody.xml')
    
    #2. Apply this classifier on the img
    pedestrians = pedestrian_cascade.detectMultiScale(img, scaleFactor = 1.05,minNeighbors = 3)
    
    #3. apply non maximum supression or not 
    if nms:
        print('Using non-maximum supression')
        pick  = non_max_suppression(pedestrians , probs=None, overlapThresh=0.2)
        print("[INFO] {}: {} original boxes, {} after suppression".format(frame_n, len(pedestrians), len(pick)))
    else:
        print('NOT Using non-maximum supression')
        pick = pedestrians.copy()
    
    #4. Draw bounding boxes
    for (x,y,w,h) in pick:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, 'Pedestrian', (x + 6, y - 6), font, 1., (0, 255, 0), 3)
     #   plt.imshow(img)
  #  plt.title('After detection {}'.format(frame_n))
  #  plt.show()
    
    #5. Print results and outputs bboxes
    print('Number of detected boxes/pedestrians:', len(pick))
    return frame_n,pick    


def pedestrians_bb(frame_n,ped_bboxes):
    """
    Transform bboxes to this format [frame_n, bb_id,x,y,dx,dy]
    ---
    input: frame_n, ped_bboxes : pedestrian bounding boxes [x,y,dx,dy]
    ---
    output: reformatted bounding boxes [frame_n, bb_id,x,y,dx,dy]
    """
    n = len(ped_bboxes)
    output = []
    if n==0 : return []
    for i in range(n):
        x = [frame_n,i+1]+list(ped_bboxes[i])
        output.append(x)
    return output


def evaluate(frame_n,solution, gt):
    """
    Evaluate the result using intersection over union score
    ---
    input: frame_n : frame number , solution : reformated bounding boxes, gt: groud truth bounding boxes
    ---
    output : final score
    """
    bbs_sol = solution
    bbs_gt = annotations_for_frame(gt, frame_n)
    black_sol = np.zeros((_H, _W))
    black_gt = np.zeros((_H, _W))
    for bb in bbs_sol:
        x, y = bb[2:4]
        dx, dy = bb[4:6]
        cv2.rectangle(black_sol, (x, y), (x+dx, y+dy), (255), -1)
    for bb in bbs_gt:
        x, y = bb[2:4]
        dx, dy = bb[4:6]
        cv2.rectangle(black_gt, (x, y), (x+dx, y+dy), (255), -1)
    # intersection over union
    intersection = black_sol * black_gt
    intersection[intersection > 0.5] = 1
    union = black_sol + black_gt
    union[union > 0.5] = 1
    score=intersection.sum()/union.sum()  
    return score

def display_detectors_results(imagePath,detector,gt_path='./gt/gt.txt',nms=True):
    """
    This function displays the final result and gives 
    ---
    input: frame_n : frame number , solution : reformated bounding boxes, gt: groud truth bounding boxes
    ---
    output : final score for a single image
    """
    if detector ==1: 
        print('Activate First detctor: HOG +SVM + OpenCV pedestrian ...')
        frame_n,pedestrians = pedestrian_detect1(imagePath,nms)
        gt = read_gt(gt_path)
        solution = pedestrians_bb(frame_n,pedestrians)
        avg_iou_score = evaluate(frame_n,solution,gt)
        print('Frame n={}, Score = {:1.3%}'.format(frame_n,avg_iou_score))
    
        
    elif detector ==2:
        print('Activate Second detctor: pretrained model ...')
        frame_n,pedestrians = pedestrian_detect2(imagePath,nms,gray=False,)
        gt = read_gt(gt_path)
        solution = pedestrians_bb(frame_n,pedestrians)
        avg_iou_score = evaluate(frame_n,solution,gt)
        print('Frame n={}, Score = {:1.3%}'.format(frame_n,avg_iou_score))
    else: 
        print('Error ...')
    return solution
  

def evaluate_on_all_images(data_root,_N,detector):
    """
    This function evaluates both detectors on the whole set of images
    ---
    input : dataroot = './folder/', _N=number of frames, detector = 1 or 2
    ---
    output : list of scores for all frames, list of solutions for all frames
    """
    gt = read_gt(gt_path)
    
    if detector ==1 :
        solutions =[]
        scores = []
        for i in tqdm.tqdm(range(_N)):
            imagePath = data_root+format_id(i+1)+'.jpg'
            print(imagePath)
            frame_n,pedestrians = pedestrian_detect1(imagePath,nms=True)
            solution = pedestrians_bb(frame_n,pedestrians)
            solutions.append(solution)
            score = evaluate(frame_n,solution,gt)
            scores.append(score)
            print('Frame n={}, Score = {:1.3%}'.format(frame_n,score))

            
    if detector ==2 :
        solutions =[]
        scores = []
        for i in tqdm.tqdm(range(_N)):
            imagePath = data_root+format_id(i+1)+'.jpg'
            print(imagePath)
            frame_n,pedestrians = pedestrian_detect2(imagePath,nms=True)
            solution = pedestrians_bb(frame_n,pedestrians)
            solutions.append(solution)
            score = evaluate(frame_n,solution,gt)
            scores.append(score)
            print('Frame n={}, Score = {:1.3%}'.format(frame_n,score))

    return scores, solutions

        
        
### main script
# example
imagePath = data_root+'300.jpg'
print('Display the results for image 300 and using detector 1 ...')
bb = display_detectors_results(imagePath,detector=1,gt_path='./gt/gt.txt',nms=True)

print('Display the results for image 300 and using detector 2 ...')
bb = display_detectors_results(imagePath,detector=2,gt_path='./gt/gt.txt',nms=True)
     
# Results on all the frames
print('Computing the results for all images(detector 1 and 2) ...')
scores1, solutions1 = evaluate_on_all_images(data_root,_N,detector=1)
scores2, solutions2 = evaluate_on_all_images(data_root,_N,detector=2)

results ={}
results[1] = {'bb': solutions1, 'scores': scores1}
results[2] = {'bb': solutions2, 'scores': scores2}

print('Save results ...')
pickle.dump(results,open('results.p','wb'))

print('Finished ...')