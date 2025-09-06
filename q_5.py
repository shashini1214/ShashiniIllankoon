import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_orig = cv.imread('a1images/shells.tif', cv.IMREAD_GRAYSCALE)

# Define a function to histogram equalization
def equalize_histogram(image):
    M, N = image.shape
    L = 256
    # Calculate the histogram of the original image
    hist_orig, bins_orig = np.histogram(image.ravel(), 256, [0, 256])
    # Calculate the CDF of the histogram
    cdf = hist_orig.cumsum()
    # Create transformation function 't' based on  CDF.
    t = np.array([(L-1) / (M*N) * cdf[i] for i in range(256)]).astype('uint8')
    img_eq = cv.LUT(image, t)
    return img_eq, hist_orig

# Apply the histogram equalization function to original image
img_eq, hist_orig = equalize_histogram(img_orig)
hist_eq, bins_eq = np.histogram(img_eq.ravel(), 256, [0, 256])

# Create a figure with a 2x2 grid of subplots to display the results
fig, ax = plt.subplots(2, 2, figsize=(12, 4))

ax[0,0].imshow(img_orig, cmap='gray', vmin=0, vmax=255) # Display as grayscale
ax[0,0].set_title('Original Image')

ax[0,1].imshow(img_eq, cmap='gray', vmin=0, vmax=255)
ax[0,1].set_title('Equalized Image')

ax[1,0].plot(hist_orig)
ax[1,0].set_title('Histogram of Original Image')

ax[1,1].plot(hist_eq)
ax[1,1].set_title('Histogram of Equalized Image')

for i in ax[0, :]:
    i.axis('off')

plt.show()