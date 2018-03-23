# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:02:38 2018

@author: firebug
"""
import numpy as np
import matplotlib.pyplot as plt
    
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

def equalization(img, cdf):
    sk = np.uint8(255 * cdf) 
    M=len(img)
    N=len(img[0])
    E = np.zeros_like(img)
    for i in range(M):
        for j in range(N):
            E[i, j] = sk[img[i, j]]

    newPdf = pdfhist(E)
    newCdf = cdfhist(newPdf)
    return E, newPdf, newCdf

img = plt.imread("hw1/index3.jpg")
plt.subplot(4,1,1)
plt.imshow(img, cmap='gray')
plt.title('Original image')
plt.axis('off')
greylevel = np.arange(256)

arr = np.array(img)

pdf = pdfhist(arr)
cdf = cdfhist(pdf)

plt.subplot(4,1,2)
plt.xlim((0,256))
plt.bar(greylevel,pdf, color='r', alpha=0.5)

plt.twinx()
plt.xlim((0,256))
plt.bar(greylevel,cdf, color='b', alpha=0.5)
plt.title('PDF(red) & CDF(blue)')


eimg, epdf, ecdf = equalization(arr, cdf)

plt.subplot(4,1,3)
plt.imshow(eimg, cmap='gray')
plt.title('Equalized image')
plt.axis('off')

plt.subplot(4,1,4)
plt.xlim((0,256))
plt.bar(greylevel,epdf, color='r', alpha=0.5)

plt.twinx()
plt.xlim((0,256))
plt.bar(greylevel,ecdf, color='b', alpha=0.5)
plt.title('Equalized PDF(red) & CDF(blue)')

plt.tight_layout()
plt.show()