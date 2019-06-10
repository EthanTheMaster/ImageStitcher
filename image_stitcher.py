import cv2
import numpy as np
from numpy import inf
from matplotlib import pyplot as plt
import argparse

def calculate_entropy(img):
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    #Calculate Informational Entropy
    histogram_prob = histogram / np.sum(histogram)
    histogram_prob_log = np.log(histogram_prob)
    #Zero out -inf values due to taking log of 0
    histogram_prob_log[histogram_prob_log == -inf] = 0.0
    return np.sum(-1 * histogram_prob * histogram_prob_log)

#Stich two colored images together ... if img1 is translated by <img1_x_trans, img1_y_trans>, then two images will align 
def stitch(img1_color, img2_color, img1_x_trans, img1_y_trans):
    #Generate empty image that will hold the composite image
    stitched_image = np.zeros((img1_color.shape[0] + abs(img1_y_trans), img1_color.shape[1] + abs(img1_x_trans), img1_color.shape[2]), np.uint8)
    stitched_height = stitched_image.shape[0]
    stitched_width = stitched_image.shape[1]
    if img1_y_trans < 0:
        #img1 is moved upwards ... place img1 on top of img2
        if img1_x_trans <= 0:
            stitched_image[stitched_height - img2_color.shape[0]:, stitched_width - img2_color.shape[1]:, 0:] = img2_color
            stitched_image[0:img1_color.shape[0], 0:img1_color.shape[1], 0:] = img1_color
        else:
            stitched_image[stitched_height - img2_color.shape[0]:, 0:img2_color.shape[1], 0:] = img2_color
            stitched_image[0:img1_color.shape[0], stitched_width - img1_color.shape[1]:, 0:] = img1_color
    else:
        #img1 is moved downwards ... place img2 on top of img1
        if img1_x_trans <= 0:
            stitched_image[stitched_height - img1_color.shape[0]:, 0:img1_color.shape[1], 0:] = img1_color
            stitched_image[0:img2_color.shape[0], stitched_width - img2_color.shape[1]:, 0:] = img2_color
        else:
            stitched_image[stitched_height - img1_color.shape[0]:, stitched_width - img1_color.shape[1]:, 0:] = img1_color
            stitched_image[0:img2_color.shape[0], 0:img2_color.shape[1], 0:] = img2_color
    return stitched_image

#n_divisions - makes a nxn grid in region of interest and creates templates from the cells
#k_template_matches - number of votes for transformations to be accepted ... higher is more accurate but at cost of performance
def calc_translation(img1_path, img2_path, n_divisions, k_template_matches):

    img1_color = cv2.imread(img1_path, 1)
    img2_color = cv2.imread(img2_path, 1)

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    #Calculate difference map to eliminate common elements then apply blur to denoise
    diff = np.abs(img2 - img1)
    diff = cv2.medianBlur(diff, 11)
    diff = cv2.medianBlur(diff, 21)
    diff = cv2.medianBlur(diff, 31)

    #Convert difference map to binary image and then generate bounding box
    ret, threshold_diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)

    roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(threshold_diff)
    print (str(roi_x) + ", " + str(roi_y) + ", " + str(roi_w) + ", " + str(roi_h))

    #Generate Entropy Graph
    #Create mask window ... divide roi into a n_divisions x n_divisions grid
    mask_width = roi_w / n_divisions
    mask_height = roi_h / n_divisions
    entropy_graph = np.zeros((n_divisions, n_divisions))

    #Slide mask to each grid cell
    for xi in range(0, n_divisions):
        for yi in range(0, n_divisions):
            #Slide mask window
            region_x = roi_x + xi * mask_width
            region_y = roi_y + yi * mask_height

            masked_image = img1[region_y:region_y + mask_height, region_x:region_x + mask_width]
            entropy_graph[yi][xi] = calculate_entropy(masked_image)

    #Sort entropy graph to get k most entropic regions
    flatted_entropy_graph = np.ndenumerate(entropy_graph)
    flatted_entropy_graph = sorted(flatted_entropy_graph, key=lambda pair: pair[1], reverse=True)
    
    #Use voting system for best translations
    votes = {}
    for ((row, col), entropy) in flatted_entropy_graph:
        region_x = roi_x + col * mask_width
        region_y = roi_y + row * mask_height
        #Create template on highly entropic section of image
        template = img1[region_y:region_y+mask_height, region_x:region_x+mask_width]

        template_match = cv2.matchTemplate(img2, template, cv2.TM_SQDIFF)
        _, _, min_loc, _ = cv2.minMaxLoc(template_match)

        print("Detected at: " + str(min_loc))
        print("Original Loc: " + str((region_x, region_y)))
        x_trans = (min_loc[0] - region_x)
        y_trans = (min_loc[1] - region_y)
        if (x_trans, y_trans) in votes:
            votes[(x_trans, y_trans)] += 1
            if votes[(x_trans, y_trans)] == k_template_matches:
                break
        else:
            votes[(x_trans, y_trans)] = 1
            if votes[(x_trans, y_trans)] == k_template_matches:
                break

    #Use translation with highest votes
    return (max(votes, key=votes.get), (roi_x, roi_y, roi_w, roi_h))

#Do CLI argument parsing
parser = argparse.ArgumentParser()

parser.add_argument("--roi", nargs="*", type=int, help="Screenshot is cropped to the specified rectangle (top left x, top left y, width, height) and if no coordinates are given, the stitcher will auto-detect a region of interest.")
parser.add_argument("--N", type=int, help="Region of interest (roi) is divided into an NxN grid and stitcher will template match on the grid cells. The default value is 15.")
parser.add_argument("--votes", type=int, help="Stitcher determines a transformation through voting, and once a transformation reaches the specified number of votes, that transformation is chosen. A higher value results in higher accuracy at the expense of performance. The default value is 7.")
parser.add_argument("--path", help="List of paths to sequence of screenshots", nargs='+')

args = parser.parse_args()

#Set up parameters for stitcher

#algorithm will initially find a region of interest and crop all subsequent images in that region
#roi calculation is very aggressive and is last resort for when there are fixed elements in screenshots
focus_roi_only = False
roi_x, roi_y, roi_w, roi_h = (0,0,0,0)
n_divisions = 15
k_template_matches = 7
shot_sequence = []

if args.path is None:
    print("Stitcher requires a list of screenshots.")
    exit()

if len(args.path) < 2:
    print("Stitching images require 2 or more images.")
    exit()
else:
    #Create user-specified region of interest
    if not(args.roi is None):
        if len(args.roi) == 4:
            roi_x, roi_y, roi_w, roi_h = (args.roi[0],args.roi[1],args.roi[2],args.roi[3])
        if len(args.roi) > 0 and len(args.roi) != 4:
            print("In order to specify a region of interest, 4 integers are required to specify a rectangle: top left x, top left y, width, height")
            exit()
        focus_roi_only = True
    if not(args.N is None):
        n_divisions = args.N
    if not(args.votes is None):
        k_template_matches = args.votes
    shot_sequence = args.path


#last_stitch holds last made composite image ... "fold" onto this image to form final image
last_stitch = []
for i in range(0, len(shot_sequence) - 1):
    img1_path = shot_sequence[i]
    img2_path = shot_sequence[i+1]

    print("path1: " + img1_path)
    print("path2: " + img2_path)

    translation, roi_rect = calc_translation(img1_path, img2_path, n_divisions, k_template_matches)

    img1_color = cv2.imread(img1_path, 1)
    img2_color = cv2.imread(img2_path, 1)

    if (roi_x, roi_y, roi_w, roi_h) == (0,0,0,0):
        roi_x, roi_y, roi_w, roi_h = roi_rect

    if focus_roi_only:
        img1_color = img1_color[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :]
        img2_color = img2_color[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :]
        
    print("Translation: " + str(translation))

    if last_stitch == []:
        last_stitch = stitch(img1_color, img2_color, translation[0], translation[1])    
    else:
        last_stitch = stitch(last_stitch, img2_color, translation[0] , translation[1])

cv2.imwrite('composite.png',last_stitch)
plt.imshow(cv2.cvtColor(last_stitch, cv2.COLOR_BGR2RGB))

plt.show()