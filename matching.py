# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:29:21 2018

@author: firebug
"""
import numpy as np
import matplotlib.pyplot as plt

# Calculate the 1D array's histogram
def hist(arr):
    h = np.zeros(256)
    M=len(arr)
    for i in range(M):
        k=arr[i]
        h[k] += 1
    for d in range(256):
        h[d] = h[d]/(M)
    return h
    
def pdfhist(img):
    h = np.zeros(256)
    M=len(img)
    N=len(img[0])
    for i in range(M):
        for j in range(N):
            k=img[i][j]
            h[k] += 1
            
    # Normilized the histogram
    for d in range(256):
        h[d] = h[d]/(M*N)
    return h
    
def cdfhist(pdf):
    H = np.zeros(256)
    for i in range(256):
        H[i] = H[i-1] + pdf[i]
    return H
    
def matching(srcimg, mapimg):
    
    imres = srcimg.copy()
    
    # Calculate by R,G,B 3 channels
    for ch in range(imsrc.shape[2]):
   
        srcarr=np.array(srcimg[:,:,ch].flatten())
    # Calculate the src image cdf
        
        srccdf = cdfhist(hist(srcarr))
    # Calculate the map image cdf
        maparr = np.array(mapimg[:,:,ch].flatten())
        mapcdf = cdfhist(hist(maparr))
 
        bins = np.arange(256)
    # Map the mapcdf to the src image
        imgtmp = np.interp(srcarr, bins, srccdf)
    
        imgtmp2=  np.interp(imgtmp, mapcdf, bins)
        imres[:,:,ch] = imgtmp2.reshape((srcimg.shape[0],srcimg.shape[1]))
    
    newarr = np.array(imres)
    newpdf = pdfhist(newarr)
    newcdf = cdfhist(newpdf)
    return imres, newpdf, newcdf
    
imsrc = plt.imread('l1.jpg')
immap = plt.imread('index4.jpg')
greylevel = np.arange(256)

newimg, pdf, cdf = matching(imsrc,immap)

plt.subplot(2,1,1)
plt.imshow(newimg)
plt.title('Original image')
plt.axis('off')
plt.subplot(2,1,2)
plt.xlim((0,256))
plt.bar(greylevel,pdf, color='r', alpha=0.5)

plt.twinx()
plt.xlim((0,256))
plt.bar(greylevel,cdf, color='b', alpha=0.5)
plt.title('PDF(red) & CDF(blue)')

plt.tight_layout()
plt.show()
