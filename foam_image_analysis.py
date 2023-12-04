# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:59:13 2023

@author: Leonardo Chiappisi (Institut Laue-Langevin)
"""

import pandas as pd
import os
import cv2   #install opencv-python
import numpy as np
import matplotlib.pyplot as plt
# from foam_image_analysis_functions import load_and_convert_image, reject_borders, find_countours, calc_PB_border_radius
# from foam_image_analysis_functions import analyse_contours, calc_liquid_volume_fraction, calc_specific_area
from foam_image_analysis_functions import *
from plot_results import plot_results
from pathlib import Path, PurePath
import gc
import sys


# path_data = (r'U:\\Uni\\PSCM Coordinator\\HDR\plots\\Foam\\data_analysis\\Image corrigées')
path_data = (r'9-10-1762/bottom')


pixel_size= 0.00567 # size of the pixel in mm 

region_of_interest = ((100,3500),(50,-300)) 


define_region_of_interest = False
# PB_analysis = False
PB_analysis = True #Is the PB border thickness determined. 
min_vol_f_for_PB = 5.0 #PB border radius will be determined for images with volume fraction lower than this one. 
NPB_fits = 50 #Number of fits to determine the PB radius


if define_region_of_interest is True:
    plot_region_of_interest(path_data,region_of_interest,2)
    sys.exit()

try:
    all_data_params = pd.read_csv(os.path.join(path_data, 'mean_data.csv'))
except:
    all_data_params = pd.DataFrame()
    
img_gray, canv, bin_img, img_bin_no_borders = None, None, None, None  #objects created outside the loop in order to reduce memory consumption.
for filename in sorted(os.listdir(path_data)):
    if filename.endswith('.png'):
        # if int(filename.split('.')[0]) == 309322:
            print(20*'*' + '\n')
            print(filename)
        
            #creates a pd.dataframe where all the parameters of a given image are stored
            average_params = pd.DataFrame()  
            #return the image in gray scale, an empty canvas, and the binarized image, corrected for spots and other noise.
            #the image is resized to take into account the sqrt(2) factor
            img_gray, canv, bin_img = load_and_convert_image(os.path.join(path_data, filename), region_of_interest)  
            #eliminates all the bubbles which are at the borders. 
            # print('image_loaded')
            img_bin_no_borders = reject_borders(bin_img) 
            # print('borders rejected')
        #     #plots and saves the binarized image
            if os.path.exists(os.path.join(path_data, 'bin_img')):
                None
            else:
                os.mkdir(os.path.join(path_data, 'bin_img'))
            fig_bin = plt.figure()
            plt.axis('off')
            plt.imshow(bin_img, cmap=('gray'))
            cv2.imwrite(os.path.join(path_data, 'bin_img') + '/' + os.path.splitext(filename)[0]+'_binary.jpg', bin_img)
            plt.close('fig_bin')
            # plt.savefig(os.path.join(path_data, 'bin_img') + '/' + os.path.splitext(filename)[0]+'_binary.jpg',dpi = 259, bbox_inches='tight', pad_inches=0)
        #     print('binary image saved') 
            #uses the cv2 contour function to extract geometrical informations on the bubbles
            # such as the areas, perimeter, liquid fraction in the foam volume (frac_v) and at the surface (vol_h), 
            contours = find_countours(img_bin_no_borders)
        #     print('contours found')
            radii, areas, perimeters, centers_of_mass = analyse_contours(contours, min_area=50)  # in pixel
            np.savetxt(os.path.join(path_data, str(filename) + '_data_radius.txt'), radii*pixel_size, header='#bubble radii in mm')
        #     print('radius calculated')
            vol_h, frac_v = calc_liquid_volume_fraction(bin_img)
            #calculates the specific area of the foam plateau borders n pixel-1
            S_V_val = calc_specific_area(bin_img)
            print(f'liquid volume fraction: {frac_v:1.1f}%, PB_length is {vol_h/S_V_val*3.:1.1f} pixels')
            
            #calculates the Sauter mean radius, the mean radius and the polydispersity
            R32 = np.sum(radii**3)/np.sum(radii**2)
            R_mean = np.average(radii)
            R_std = np.std(radii)
            R_PI = R_std/R_mean #polidispersity index
            N_bubble = len(radii)
            # print('parameters of bubbles obtained') 
            if PB_analysis is True:
                if frac_v < min_vol_f_for_PB:    
                    PB_bin_radius, PB_bin_radius_std, PB_gauss_radius, PB_gauss_radius_std, PB_attemps = calc_PB_border_radius(NPB_fits, contours, img_gray, bin_img, filename, path_data, vol_h/S_V_val*3.)
                    plt.close("all")
                else:
                    PB_bin_radius, PB_bin_radius_std, PB_gauss_radius, PB_gauss_radius_std, PB_attemps = np.nan, np.nan, np.nan, np.nan, 0
                    print('PB calculated')
            else:
                PB_bin_radius, PB_bin_radius_std, PB_gauss_radius, PB_gauss_radius_std, PB_attemps = np.nan, np.nan, np.nan, np.nan, 0
            
            
            if PB_attemps < NPB_fits*0.75:
                print(f'PB border thickness was determined for {PB_attemps} borders only')
            print(f'''PB_bin_radius: {PB_bin_radius},
        PB_bin_radius_std: {PB_bin_radius_std},
        PB_gauss_radius: {PB_gauss_radius},
        PB_gauss_radius_std: {PB_gauss_radius_std},
        PB_attemps: {PB_attemps}''')
                
            data_pars = {'file': filename,
                          'S_V_img / cm-1': S_V_val / pixel_size * 10, # cm^(-1)
                          'radius_mean / mm' : R_mean * pixel_size, #mm 
                          'Sauter mean radius / mm' : R32 * pixel_size, #mm
                          'radius_std / mm' : R_std * pixel_size, #mm
                          'polydispersity index' : R_PI,
                          'liquid_frac / %' : frac_v,
                          'N_bubbles' : N_bubble,
                          'PB_radius_bin / um': PB_bin_radius * pixel_size * 1000, #in um
                          'PB_radius_bin_std / um': PB_bin_radius_std * pixel_size * 1000, #in um
                          'PB_radius_gauss / um': PB_gauss_radius * pixel_size * 1000, #in um
                          'PB_radius_gauss_std / um': PB_gauss_radius_std * pixel_size * 1000, #in um                 
                          'PB_attemps': PB_attemps,
                          } # %}
        
            
            all_data_params = all_data_params.append(data_pars, ignore_index=True)
            all_data_params.set_index('file', inplace=False).to_csv(os.path.join(path_data, 'mean_data.csv'))
        
    
    
        
            del img_gray, canv, bin_img, img_bin_no_borders
            gc.collect()


all_data_params = all_data_params.set_index('file')
all_data_params.to_csv(os.path.join(path_data, 'mean_data.csv'))
plot_results(all_data_params, path_data)
