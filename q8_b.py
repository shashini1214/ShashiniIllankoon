import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_small = cv.imread('a1images/a1q5images/im01small.png')
img_large = cv.imread('a1images/a1q5images/im01.png')

def zoom(image, factor, interpolation_type):
    if(interpolation_type == 'nearest'):
        return cv.resize(image, None, fx=factor, fy=factor, interpolation=cv.INTER_NEAREST)
    elif(interpolation_type == 'bilinear'):
        return cv.resize(image, None, fx=factor, fy=factor, interpolation=cv.INTER_LINEAR)

s = 4
# Nearest neighbor interpolation   
img_zoomed_nearest = zoom(img_small, s, 'nearest')

img_large_resized = cv.resize(img_large, (img_zoomed_nearest.shape[1], img_zoomed_nearest.shape[0]))
ssd_n = np.sum((img_zoomed_nearest.astype(np.float32) - img_large_resized.astype(np.float32)) ** 2)
ssd_n_normalized = ssd_n / (img_large.shape[0] * img_large.shape[1])

# Bilinear interpolation
img_zoomed_bilinear = zoom(img_small, s, 'bilinear')

img_large_resized = cv.resize(img_large, (img_zoomed_bilinear.shape[1], img_zoomed_bilinear.shape[0]))
ssd_b = np.sum((img_zoomed_bilinear.astype(np.float32) - img_large_resized.astype(np.float32)) ** 2) 
ssd_b_normalized = ssd_b / (img_large.shape[0] * img_large.shape[1])

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(cv.cvtColor(img_large, cv.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[1].imshow(cv.cvtColor(img_zoomed_nearest, cv.COLOR_BGR2RGB))
ax[1].set_title(f'Zoomed Image (Nearest Neighbor)\nSSD: {ssd_n_normalized:.4f}')
ax[2].imshow(cv.cvtColor(img_zoomed_bilinear, cv.COLOR_BGR2RGB))
ax[2].set_title(f'Zoomed Image (Bilinear)\nSSD: {ssd_b_normalized:.4f}')
for a in ax:
    a.axis('off')       
plt.show()