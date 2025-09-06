import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

img_orig = cv.imread('a1images/einstein.png', cv.IMREAD_GRAYSCALE)

  # (a) Using filter2D
sobel_x_2d = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y_2d = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

im_x_2d = cv.filter2D(img_orig, cv.CV_64F, sobel_x_2d)
im_y_2d = cv.filter2D(img_orig, cv.CV_64F, sobel_y_2d)

# (b) Writing your own code for Sobel filtering
def filter_custom(image, kernel):
  k_hh, k_hw = math.floor(kernel.shape[0]/2), math.floor(kernel.shape[1]/2)
  h, w = image.shape
  image_float = cv.normalize(image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
  result = np.zeros(image.shape, 'float')

  for m in range(k_hh, h - k_hh):
    for n in range(k_hw, w - k_hw):
      result[m,n] = np.dot(image_float[m-k_hh:m + k_hh + 1, n - k_hw : n + k_hw + 1].flatten(), kernel.flatten())
  return result

im_x_custom = filter_custom(img_orig, sobel_x_2d)
im_y_custom = filter_custom(img_orig, sobel_y_2d)

# (c) Using separable filters
array1 = np.array([[1, 2, 1]])
array2 = np.array([[1, 0, -1]])

im_x_intermediate = cv.filter2D(img_orig, cv.CV_64F, array1.reshape(3, 1))
im_x_separable = cv.filter2D(im_x_intermediate, cv.CV_64F, array2)
im_y_intermediate = cv.filter2D(img_orig, cv.CV_64F, array2.reshape(3, 1))
im_y_separable = cv.filter2D(im_y_intermediate, cv.CV_64F, array1)


# Display results
fig, axes = plt.subplots(3, 2, figsize=(10, 15))

axes[0, 0].imshow(im_x_2d, cmap='gray')
axes[0, 0].set_title('Sobel X (filter2D)')
axes[0, 0].axis('off')

axes[0, 1].imshow(im_y_2d, cmap='gray')
axes[0, 1].set_title('Sobel Y (filter2D)')
axes[0, 1].axis('off')

axes[1, 0].imshow(im_x_custom, cmap='gray')
axes[1, 0].set_title('Sobel X (Custom)')
axes[1, 0].axis('off')

axes[1, 1].imshow(im_y_custom, cmap='gray')
axes[1, 1].set_title('Sobel Y (Custom)')
axes[1, 1].axis('off')

axes[2, 0].imshow(im_x_separable, cmap='gray')
axes[2, 0].set_title('Sobel X (Separable Filters)')
axes[2, 0].axis('off')

axes[2, 1].imshow(im_y_separable, cmap='gray')
axes[2, 1].set_title('Sobel Y (Separable Filters)')
axes[2, 1].axis('off')

plt.tight_layout()
plt.show()