import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

t1 = np.arange(51, dtype='uint8')
t2 = np.linspace(101, 255, 150-51+1).astype('uint8')
t3 = np.arange(151, 256, 1, dtype='uint8')

t = np.concatenate((t1, t2, t3), axis=0).astype('uint8')
# print(len(t))

img_orig = cv.imread('a1images/emma.jpg', cv.IMREAD_GRAYSCALE)
image_transformed = cv.LUT(img_orig, t)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img_orig, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Original Image')
ax[1].imshow(image_transformed, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Transformed Image')
for a in ax[0:2]:
    a.axis('off')
ax[2].plot(t)
ax[2].set_ylim([0, 255])
ax[2].set_xlim([0, 255])
ax[2].set_title('Transformation Function')
plt.show()