
"""
Created on Mon Nov 15 09:34:54 2021

@author: lamolinairie
"""
import csv
import matplotlib.colors as mcolors
from PIL import Image
import pandas as pd
import cv2   #install opencv-python
import numpy as np
import matplotlib.pyplot as plt
import os
import statistics

##### function to load and convert image
def load_and_convert_image(input_image):
    ''' load the file and returns it in grayscale
    as well as an empty canvas, and the binarized image'''
    # Reading image and converting it to gray scale
    # font = cv2.FONT_HERSHEY_COMPLEX
    img = cv2.imread(input_image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img[0:782, 10:582]
    
    # get a blank canvas for drawing contour
    canvas = np.zeros(gray_img.shape, np.uint8)
    
    th2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 5)
    #Couper les bords qui gènent à l exclude borders
    #threshold = threshold[0:782, 15:582]
    th2 = cv2.bitwise_not(th2)

    contours, hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(th2,[cnt],0,255,-1)
    im_adapt = cv2.bitwise_not(th2)
    im_adapt = cv2.bitwise_not(im_adapt)

    cv2.waitKey(0)

    return gray_img, canvas, im_adapt

##### function to analyse the contours and extract the area of the bubbles (and so the radius)
def analyse_contours(contours, min_area=10):
    ''' Function which analyses the contour and extract the geometrical informations, such as:
        Area, Perimeter, center of mass, ....'''
    areas = np.array([cv2.contourArea(i)for i in contours])
    perimeters = np.array([cv2.arcLength(i,True)for i in contours])
    #discards too small droplets, value for min_area should eventually be better defined
    perimeters = perimeters[areas>min_area]
    areas = areas[areas>min_area]
    centers_of_mass = []
    for j, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > min_area:
            M = cv2.moments(cnt)
            cx= int(M['m10']/M['m00'])
            cy= int(M['m01']/M['m00'])
            centers_of_mass.append([cx,cy])
    return areas, perimeters, np.array(centers_of_mass)

### Function to reject non entire bubbles (at the borders)
def reject_borders(image_):
    out_image = image_.copy()
    h, w = image_.shape[:2]
    for row in range(h):
        if out_image[row, 0] == 255:
            cv2.floodFill(out_image, None, (0, row), 0)
        if out_image[row, w - 1] == 255:
            cv2.floodFill(out_image, None, (w - 1, row), 0)
    for col in range(w):
        if out_image[0, col] == 255:
            cv2.floodFill(out_image, None, (col, 0), 0)
        if out_image[h - 1, col] == 255:
            cv2.floodFill(out_image, None, (col, h - 1), 0)
    return out_image

##### function to calculate the liquid fraction (with the numbers of black and white pixels)
def calc_liquid_volume_fraction(threshold):
    ''' From the binarized and holes filled image calculates the liquid volume fraction'''
    vol_h = 1 - np.count_nonzero(im_adapt) / (np.shape(im_adapt)[0] * np.shape(im_adapt)[1]) 
    frac_v = 100*0.36*(1-np.sqrt(1-vol_h))**2
    return vol_h, frac_v

def skeleton(img):
    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    img = cv2.bitwise_not(img)
    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break
    skel2 = cv2.bitwise_not(skel)
    cv2.waitKey(0)
    return skel, skel2

def calc_specific_area(skelete):
    ''' From the binarized and holes filled image calculates the liquid volume fraction'''
    height, width = skelete.shape
    S_V = np.count_nonzero(skel)/( height* width*0.001168)
    return S_V


path_data = r'data'

all_data_params = pd.DataFrame()

for filename in os.listdir(path_data):
    if filename.endswith('.png'):
        average_params = pd.DataFrame()
        gray_img, canvas, im_adapt = load_and_convert_image(os.path.join(path_data,filename)) #to calcultate S/V, liquid fraction
        img_out = reject_borders(im_adapt) #to calculate the radius 
       # img_out=im_adapt

        skel, skel2=skeleton(im_adapt)
        contours, hierarchy= cv2.findContours(img_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas, perimeters, centers_of_mass = analyse_contours(contours, min_area=10)
        vol_h, frac_v=calc_liquid_volume_fraction(im_adapt)
        S_V_val = calc_specific_area(skel)
        plt.figure() 
            
        plt.axis('off')
        plt.imshow(im_adapt, cmap='gray')
        plt.savefig(str(filename)+'binary.jpg',dpi = 259,bbox_inches='tight', pad_inches=0)
        

        plt.axis('off')
        plt.imshow(skel2, cmap='gray')

        plt.savefig(str(filename)+'skeleton.jpg',dpi = 260,bbox_inches='tight', pad_inches=0)
        

        rad_px= np.sqrt(areas/np.pi)
        radius =  rad_px*0.01168
        mean_radius_val = statistics.mean(radius)
        radiuslist = radius.tolist()  
        N_bubble = len(radiuslist)
        arealist=areas.tolist()
       # print(radiuslist)
       
       
       
        data_pars = {'file': filename,
                    'S_V_img': S_V_val, # cm^(-1)
                    'radius_mean' : mean_radius_val, # cm 
                    'liquid_frac' : frac_v,
                    'N_bubbles' : N_bubble} # %
        all_data_params = all_data_params.append(data_pars, ignore_index=True)
            
 #       averaged_pars = {'radius' : radiuslist }
 #       average_params = average_params.append(averaged_pars,  ignore_index=True)
       # print(average_params)
            

        with open(str(filename)+'data_radius.csv', 'w') as f:
            w = csv.writer(f, lineterminator='\n')
            for item in radiuslist:
                w.writerow([item])
  
    
  
all_data_params.set_index('file', inplace=True)
print(all_data_params)
all_data_params.to_csv('mean_data.csv')
all_data_params.fillna(value=np.nan, inplace=True)    






