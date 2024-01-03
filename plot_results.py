# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:11:35 2023

@author: Leonardo Chiappisi (Institut Laue-Langevin)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = path_data = ('images/')
data = pd.read_csv(os.path.join(path,'mean_data.csv'))
# print(data)

def plot_results(data, path):
    fig, axs = plt.subplots(3,2,sharex=True, figsize=(7,5))
    axs[0,0].scatter(data.index, data['N_bubbles'])
    axs[0,0].set_ylabel('N$_{bubbles}$')
    
    axs[1,0].scatter(data.index, data['S_V_img / cm-1'])
    axs[1,0].set_ylabel('S/V / cm$^{-1}$')
    
    axs[0,1].scatter(data.index, data['Sauter mean radius / mm'])
    axs[0,1].set_ylabel('$<R_{32}>$ / mm')
    
    axs[1,1].scatter(data.index, data['radius_mean / mm'])
    axs[1,1].errorbar(data.index, data['radius_mean / mm'], yerr=data['radius_std / mm'], alpha=0.5)
    axs[1,1].set_ylabel('$<R>$ / mm')
    
    axs[2,1].scatter(data.index, data['polydispersity index'])
    axs[2,1].set_ylabel('PDI')
    
    
    axs[2,0].scatter(data.index, data['PB_radius_gauss / um'])
    axs[2,0].errorbar(data.index, data['PB_radius_gauss / um'], yerr=data['PB_radius_gauss_std / um'], alpha=0.5)
    
    axs[2,0].scatter(data.index, data['PB_radius_bin / um'])
    axs[2,0].errorbar(data.index, data['PB_radius_bin / um'], yerr=data['PB_radius_bin_std / um'], alpha=0.5)
    
    axs[2,0].set_ylabel('PB Radius / $\mu m$')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'summary.pdf'))
    plt.show()
    
    
if __name__ == "__main__":
    plot_results(data, path)
