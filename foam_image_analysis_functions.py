# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:59:42 2023

@author: Leonardo Chiappisi (Institut Laue-Langevin)
"""

import pandas as pd
import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
import cv2   #install opencv-python
from skimage.draw import line
from lmfit import minimize, Parameters, fit_report
import random, math
import matplotlib.gridspec as gridspec
import os

def resize(img):
    img_resized = cv2.resize(img, None, fx=1.41, fy=1.0)
    return img_resized
    

##### function to load and convert image
def load_and_convert_image(input_image,ROI):
    ''' load the file and returns it in grayscale
    as well as an empty canvas, and the binarized image.
    INPUT:
    input_image: source image to be processed.
    ###########
    OUTPUT:
    gray_img: image, converted in gray_scale.
    canvas: blank canvas with the same size of the original image
    im_adapt: the binarized image, with spots cleaned. 
    '''
    # Reading image and converting it to gray scale
    # font = cv2.FONT_HERSHEY_COMPLEX
    img = cv2.imread(input_image)[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1]]

    img = resize(img)

    # print('image loaded and resized')
    # plt.imshow(img, cmap=('gray'))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.bilateralFilter(gray_img, 15, 75, 75)
    # plt.imshow(gray_img, cmap='gray')
   # gray_img = gray_img[0:782, 10:823]
    # print('image converted in black and white')
    # get a blank canvas for drawing contour
    canvas = np.zeros(gray_img.shape, np.uint8)
    
    # th2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=55, C=0)
    # th2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=55, C=0)
    ret2,th2 = cv2.threshold(cv2.GaussianBlur(gray_img,(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #Couper les bords qui gènent à l exclude borders
    #threshold = threshold[0:782, 15:582]
    # print('threshold calculated')
    # th2 = cv2.bitwise_not(th2)
    plt.axis('off')
    plt.imshow(th2, cmap='gray')
    cv2.imwrite('th2.png', th2)
    # plt.savefig('th2.jpg')
    # th2 = cv2.bilateralFilter(th2, d=5, )
    contours, hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # print('contour found')
    for cnt in contours:
        cv2.drawContours(th2,[cnt],0,255,-1)  #Fills the contour, i.e., eliminates the spots within the bubbles.
    
    contours, hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print(f'Found {len(contours)} bubbles')
    im_adapt = th2
    # im_adapt = cv2.bitwise_not(th2)
    # im_adapt = cv2.bitwise_not(im_adapt)
    cv2.imwrite('th2.png', im_adapt)
    # cv2.waitKey(0)

    return gray_img, canvas, im_adapt



##### function to analyse the contours and extract the area of the bubbles (and so the radius)
def analyse_contours(contours, min_area=10):
    ''' Analyse the contours of the image.

    Parameters
    ----------
    contours : list
        A list of contours.
    min_area : int
        The minimum area of a contour to be considered.
    
    Returns
    -------
    radii : numpy.ndarray
        An array of radii of the contours.
    areas : numpy.ndarray
        An array of areas of the contours.
    perimeters : numpy.ndarray
        An array of perimeters of the contours.
    centers_of_mass : numpy.ndarray
        An array of centers of mass of the contours.
    '''
    areas = np.array([cv2.contourArea(i)for i in contours])
    perimeters = np.array([cv2.arcLength(i,True)for i in contours])
    #discards too small droplets, value for min_area should eventually be better defined
    perimeters = perimeters[areas>min_area]
    areas = areas[areas>min_area]
    radii = np.sqrt(areas/np.pi)
   # print(type(areas))
   # circularity = np.divide(4*np.pi*areas, perimeters)
    centers_of_mass = []
    for j, cnt in enumerate(contours):
       # print(circularity)
        #if ((cv2.contourArea(cnt) > min_area) and (circularity > 0.7)) :
        if cv2.contourArea(cnt) > min_area :
            circularity = (4*np.pi*cv2.contourArea(cnt)/ pow(cv2.arcLength(cnt, True),2))
            #print(circularity)
            if circularity > 0.6 : 
                M = cv2.moments(cnt)
           # print(type(cv2.contourArea(cnt)))
                cx= int(M['m10']/M['m00'])
                cy= int(M['m01']/M['m00'])
                centers_of_mass.append([cx,cy])
    return radii, areas, perimeters, np.array(centers_of_mass)

### Function to reject non entire bubbles (at the borders)
def reject_borders(image_):
    """
    Rejects the borders of an image.
    
    Parameters
    ----------
    image_ : numpy.ndarray
        The image to reject the borders of.
    
    Returns
    -------
    numpy.ndarray
        The image with the borders rejected.
    """
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
def calc_liquid_volume_fraction(bin_img):
    """
Calculates the liquid volume fraction of a binary image.

Parameters
----------
bin_img : ndarray
    A binary image.

Returns
-------
vol_h : float
    The volume fraction of the liquid phase.
frac_v : float
    The volume fraction of the liquid phase, corrected for the
    shape of the particles.
    See Soft Matter, 2016, 12 , 8025 —8029  for further detail
    """
    vol_h = 1 - np.count_nonzero(bin_img) / (np.shape(bin_img)[0] * np.shape(bin_img)[1]) 
    frac_v = 100*0.36*(1-np.sqrt(1-vol_h))**2
    return vol_h, frac_v


def skeleton(img):
    '''
    This function takes a binary image as input and returns a skeletonized version of it.
    The algorithm is based on the paper "Skeletonization using Zone Filling" by T.Y. ZHANG and C. Y. SUEN.
    The algorithm is implemented in 4 steps:
    1. Create an empty skeleton
    2. Open the image
    3. Subtract open from the original image
    4. Erode the original image and refine the skeleton
    5. If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    The function returns the skeletonized image.
    '''
    # Step 1: Create an empty skeleton
    # size = np.size(img)
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
    #cv2.waitKey(0)
    return skel, skel2

def calc_specific_area(bin_img):
    ''' From the binarized and holes filled image calculates the liquid volume fraction'''
    
    skelete, _ = skeleton(bin_img)
    height, width = skelete.shape
    S_V = np.count_nonzero(skelete)/( height* width) #cm-1
    return S_V #in cm-1

def find_countours(img):
    '''
    Finds the contours of the image.
    
    Parameters
    ----------
    img : numpy.ndarray
        The image to find the contours of.
    
    Returns
    -------
    contours : list
        A list of contours.
    '''
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours


def calc_PB_border_radius(N, contours, img_gray, bin_img, filename, path, l):
    ''' Calculates the plateau border thickness (radius) from an image.
    N: number of evaluated points
    contours: from the cv2 contour extraction
    l = length of the analysed segmentm in pixels
    img_gray: image in black and white
    bin_img: binarized image
    '''

    maxiter = 100*N  #max attemps to perform the analysis of the PB borders. 
    def gauss(x, H, A, x0, sigma):
        return H + A * np.exp(-((x - x0) ** 2 / (2 * sigma ** 2)))

    def f2min(params):
        x=np.arange(len(rr))
        vals = params.valuesdict()
        model = vals['H'] + vals['A'] * np.exp(-(x - vals['x0']) ** 2 / (2 * vals['sigma'] ** 2))
        res = (z - model)
        # print(type(res), res)
        return res[~np.isnan(res)]
    
    def plot(ax, p1, p2, a, b, rr, cc):
        ax.imshow(bin_img, origin='upper')
        ax.set_xlim(p2[0]-100,p2[0]+100)
        ax.set_ylim(p2[1]-100,p2[1]+100)
        ax.scatter(a[0], a[1])
        ax.scatter(b[0], b[1])
        ax.scatter(rr, cc, alpha=0.25)
        ax.axis('off')
        
    def plot_gaussian(ax, x, z, z_bin, bin_width, out, fit_params):
                vals = fit_params.valuesdict()
                ax.plot(x, z)
                ax.plot(x, z_bin, label='width = {} pixel'.format(bin_width))  
                fit_gauss = gauss(x, out.params['H'].value, out.params['A'].value, out.params['x0'].value, out.params['sigma'].value)
                # print('x is:', x)
                initial_guess = gauss(x, vals['H'], vals['A'], vals['x0'], vals['sigma'])
                ax.plot(x, fit_gauss, label='width = {:.2f} pixel'.format(abs(2.355*out.params['sigma'].value)))
                ax.plot(x, initial_guess, label='guessed')
                ax.legend()
            
        

    widths_bin = []
    widths_gauss = []
    
    
    
    #### Creating the image where all the analyses of the plateau border thickness are stored
    rows = int(np.ceil(np.sqrt((N+1)/2)))
    cols = int(np.ceil((N+1)/rows)) if rows > 0 else 1
    gs = gridspec.GridSpec(rows, cols)
    
    fig = plt.figure(figsize=(cols*3.5, rows*3.5))
    fig2 = plt.figure(figsize=(cols*3.5, rows*3.5))
    
    _ = 0
    __= -1
    while _ < N and __ < maxiter:
        # print(20*'*', '\n', _, __,  '\n', 20*'*')
        __ += 1
        try:
            contour_number = random.randint(0,len(contours)-1)
            points = random.randint(0,len(contours[contour_number]-4))

       
            p1 = contours[contour_number][points][0]
            p2 = contours[contour_number][points+3][0]
       
       

            if p2[0] == p1[0]:
                slope = 0
            elif p2[1] == p1[1]:
                slope = np.inf
            else:
                slope = -1/((p2[1]-p1[1])/(p2[0]-p1[0]))
            

            angle = math.atan(slope)
            mid_point = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
       
       
            a = (mid_point[0]+int(round(l*math.cos(angle))), mid_point[1]+int(round(l*math.sin(angle))))
            b = (mid_point[0]-int(round(l*math.cos(angle))), mid_point[1]-int(round(l*math.sin(angle))))
           
            rr, cc = line(a[0],a[1], b[0],b[1])  #coordinates of pixels connecting point a and b
       

     
       

            z = []  #list in which thickness are stored, determined upon gaussian fit of non_binarized image
            z_bin = [] #list of thickness from binarized images
       
       
            for i in range(len(rr)):
                # print('profile', rr[i], cc[i], img_gray[cc[i], rr[i]] )
                z.append(img_gray[cc[i], rr[i]])
                z_bin.append(bin_img[cc[i], rr[i]])
           
            z_bin = np.array(z_bin)
            bin_width = len(z_bin[z_bin[:]==0]) #count the number of pixels with z=0

       
            
            fit_params = Parameters()
            # print('A will be:', np.amin(z), np.amax(z), float(np.amin(z))-float(np.amax(z)), np.argmin(z) )
            fit_params.add('H', value = np.amax(z), vary=True, min=160, max = 240)
            fit_params.add('A', value = float(np.amin(z))-float(np.amax(z)), vary=True, min=-200, max=-30)
            fit_params.add('x0', value = float(np.argmin(z)), vary=True)
            fit_params.add('sigma', value = l/3./2.3, min=l/20., max=l, vary=True)#, min=1, max=10, vary=True)
           
            
            out = minimize(f2min, fit_params, method='leastsq')
            # print(fit_report(out))
            fit_gauss = gauss(range(len(rr)), out.params['H'].value, out.params['A'].value, out.params['x0'].value, out.params['sigma'].value)

            if out.redchi >= 100:
            #     ax = fig2.add_subplot(gs[_])
            #     plot_gaussian(ax, range(len(rr)), z, z_bin, bin_width, out, fit_params)
                raise Exception()  #if the reduced chi square is too high, ignore this attempt and restart with the next image
            
            
            ax = fig.add_subplot(gs[_])
            plot(ax, p1, p2, a, b, rr, cc)

       
            widths_bin.append(bin_width)
            widths_gauss.append(abs(2.355*out.params['sigma'].value))
            
            ax = fig2.add_subplot(gs[_])
            plot_gaussian(ax, np.arange(len(rr)), z, z_bin, bin_width, out, fit_params)

            _ += 1
        except:
            None
    PB_calculations = _        
    gs.tight_layout(fig)
    fig.savefig(os.path.join(path, '{}_PB_section.pdf'.format(filename)))
    
    ax = fig2.add_subplot(gs[_+1])
    ax.hist(widths_bin, 10, label='bin, mean = {:.1f} pixel'.format(np.average(widths_bin)), alpha=0.5)
    ax.hist(widths_gauss, 10, label='Gauss, mean = {:.1f} pixel'.format(np.average(widths_gauss)), alpha=0.5)
    
    gs.tight_layout(fig2)
    fig2.savefig(os.path.join(path, '{}_PB_fits.pdf'.format(filename)))
    plt.close('all')    
    
    
    width_bin_average, width_bin_std = np.average(widths_bin)/2, np.std(widths_bin)/2
    width_gauss_average, width_gauss_std = np.average(widths_gauss)/2, np.std(widths_gauss)/2
    
    del fig, fig2
    return width_bin_average, width_bin_std, width_gauss_average, width_gauss_std, PB_calculations

