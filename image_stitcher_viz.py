import cv2
import numpy as np
from numpy import inf
from matplotlib import pyplot as plt

folder_path = "/home/ethanlam/Pictures/Screenshots/Sample1/"
shot_sequence = ["Shot1.png", "Shot2.png", "Shot3.png", "Shot4.png", "Shot5.png"]

img0 = cv2.imread((folder_path + shot_sequence[0]), 0)
img1 = cv2.imread((folder_path + shot_sequence[2]), 0)
img0_color = cv2.imread((folder_path + shot_sequence[2]), 1)
img1_color = cv2.imread((folder_path + shot_sequence[2]), 1)

def calculate_entropy(img):
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    #Calculate Informational Entroy
    histogram_prob = histogram / np.sum(histogram)
    histogram_prob_log = np.log(histogram_prob)
    #Zero out -inf values due to taking log of 0
    histogram_prob_log[histogram_prob_log == -inf] = 0.0
    return np.sum(-1 * histogram_prob * histogram_prob_log)

def generate_mask(mask_size_x, mask_size_y, mask_x, mask_y):
    mask = np.zeros(img0.shape, np.uint8)
    mask[mask_y:mask_y+mask_size_y, mask_x:mask_x+mask_size_x] = 255
    return mask

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

#Calculate difference map to eliminate common elements and create region of interest to do tracking in then apply blur to denoise
diff = np.abs(img1 - img0)
diff = cv2.medianBlur(diff, 11)
diff = cv2.medianBlur(diff, 21)
diff = cv2.medianBlur(diff, 31)

ret, threshold_diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)

roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(threshold_diff)
print (str(roi_x) + ", " + str(roi_y) + ", " + str(roi_w) + ", " + str(roi_h))

#Generate Entroy Graph
#Create mask window
n = 15
mask_width = roi_w / n
mask_height = roi_h / n
entropy_graph = np.zeros((n, n))

for xi in range(0, n):
    for yi in range(0, n):
        #Slide mask window
        region_x = roi_x + xi * mask_width
        region_y = roi_y + yi * mask_height

        masked_image = img0[region_y:region_y + mask_height, region_x:region_x + mask_width]
        entropy_graph[yi][xi] = calculate_entropy(masked_image)

#Sort entropy graph to get k most entropic regions
flatted_entropy_graph = np.ndenumerate(entropy_graph)
flatted_entropy_graph = sorted(flatted_entropy_graph, key=lambda pair: pair[1], reverse=True)
k = 0
k_template_matches = 7
cross_check_squared_error_threshold = 10.0
votes = {}
for ((row, col), entropy) in flatted_entropy_graph:
    region_x = roi_x + col * mask_width
    region_y = roi_y + row * mask_height
    #Create template on highly entropic section of image
    template = img0[region_y:region_y+mask_height, region_x:region_x+mask_width]

    template_match = cv2.matchTemplate(img1, template, cv2.TM_SQDIFF)
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

x_trans, y_trans = max(votes, key=votes.get)
print("Translation: " + str((x_trans, y_trans)))
        
stitched_image = stitch(img0_color, img1_color, x_trans, y_trans)
print(img0_color.shape)
print(stitched_image.shape)

#Graph Everything
plt.subplot(3, 2, 1)
plt.imshow(img0, 'gray')
plt.scatter([roi_x, roi_x + roi_w, roi_x, roi_x + roi_w], [roi_y, roi_y + roi_h, roi_y + roi_h, roi_y])

plt.subplot(3, 2, 2)
plt.imshow(img1, 'gray')
plt.scatter([roi_x, roi_x + roi_w, roi_x, roi_x + roi_w], [roi_y, roi_y + roi_h, roi_y + roi_h, roi_y])

plt.subplot(3, 2, 3)
plt.imshow(threshold_diff, 'gray')

plt.subplot(3, 2, 4)
plt.imshow(entropy_graph, 'gray')

plt.subplot(3, 2, 4)
plt.imshow(entropy_graph, 'gray')

plt.subplot(3, 2, 5)
plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))

plt.show()