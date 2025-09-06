import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_orig = cv.imread('a1images/jeniffer.jpg')

# Convert original image from BGR color space to HSV.
img_hsv = cv.cvtColor(img_orig, cv.COLOR_BGR2HSV)

# Split the HSV image into its individual channels: h,s,v.
h, s, v = cv.split(img_hsv)

_, mask = cv.threshold(s, 13, 255, cv.THRESH_BINARY)  # Apply thresholding to saturation plane

foreground = cv.bitwise_and(img_orig, img_orig, mask=mask)  # Extract foreground using mask
background = cv.bitwise_and(img_orig, img_orig, mask=cv.bitwise_not(mask))  # Extract background

# Convert extracted foreground to HSV color space to work with its Value channel.
foreground_hsv = cv.cvtColor(foreground, cv.COLOR_BGR2HSV)
h_foreground, s_foreground, v_foreground = cv.split(foreground_hsv)
hist, bins = np.histogram(v_foreground[mask > 0].ravel(), 256, [0, 256])  # Histogram of value plane of foreground
cdf = hist.cumsum()  # Cumulative distribution function

L = 256
# Get the number of pixels in foreground.
MN = v_foreground[mask > 0].size

# Create the transformation function 't' for histogram equalization, based on the foreground's CDF.
t = np.array([(L-1) / (MN) * cdf[i] for i in range(256)]).astype('uint8')

# Apply histogram equalization transformation 't' to the foreground's Value channel.
v_foreground_eq = v_foreground.copy()
# Apply transformation only to foreground pixels using mask.
v_foreground_eq[mask > 0] = t[v_foreground[mask > 0]]

# Merge the original H and S channels of foreground with equalized Value channel.
foreground_eq = cv.merge((h_foreground, s_foreground, v_foreground_eq))
foreground_eq = cv.cvtColor(foreground_eq, cv.COLOR_HSV2BGR)  # Convert back to BGR color space
img_modified = cv.add(foreground_eq, background)  # Combine modified foreground with background

# Create a figure with 3 subplots to show mask, original image, and modified image.
fig, axis = plt.subplots(1, 3, figsize=(12, 8))

# Display foreground mask.
axis[0].imshow(mask, cmap='gray')
axis[0].set_title('Foreground Mask')

# Display original image.
axis[1].imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
axis[1].set_title('Original Image')

# Display  image with equalized foreground.
axis[2].imshow(cv.cvtColor(img_modified, cv.COLOR_BGR2RGB))
axis[2].set_title('Image with Equalized Foreground')

for i in axis:
    i.axis('off')


# ----- Plotting the histograms of the value plane -----
hist_eq, bins_eq = np.histogram(v_foreground_eq[mask > 0].ravel(), 256, [0, 256])

# Create a new figure with 2 subplots to compare Value plane histograms
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Plot the histogram of original foreground Value plane.
ax[0].plot(hist)
ax[0].set_title('Histogram of Value Plane of Foreground')
ax[0].set_xlabel('Intensity Value')
ax[0].set_ylabel('Frequency')

# Plot the histogram of equalized foreground Value plane.
ax[1].plot(hist_eq)
ax[1].set_title('Histogram of Equalized Value Plane of Foreground')
ax[1].set_xlabel('Intensity Value')
ax[1].set_ylabel('Frequency')


# ----- Plotting the foreground isolated image -----
# Create a new figure with 1 subplot to show just extracted foreground.
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# Display the foreground image. Convert from BGR to RGB.
ax.imshow(cv.cvtColor(foreground, cv.COLOR_BGR2RGB))
ax.axis('off') # Turn off the axes


# ----- Plotting the 3 planes of HSV -----
# Create a new figure with 3 subplots to show the H, S, and V channels of the original image.
fig, ax = plt.subplots(1, 3, figsize=(12,4))
ax[0].imshow(h, cmap='gray') # Display Hue channel (grayscale)
ax[0].set_title('Hue')
ax[1].imshow(s, cmap='gray') # Display Saturation channel (grayscale)
ax[1].set_title('Saturation')
ax[2].imshow(v, cmap='gray') # Display Value channel (grayscale)
ax[2].set_title('Value')
for i in ax:
    i.axis('off')

plt.show()

img_orig = cv.imread('a1images/einstein.png', cv.IMREAD_GRAYSCALE)

# Create the 2D Sobel kernel for x-direction
sobel_kernel_x_2d = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=np.float32)

# Create the 2D Sobel kernel for the y-direction
sobel_kernel_y_2d = np.array([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]], dtype=np.float32)

# Convolve loaded grayscale image with x-direction Sobel kernel
sobel_x_2d = cv.filter2D(img_orig, cv.CV_32F, sobel_kernel_x_2d)

# Convolve loaded grayscale image with y-direction Sobel kernel
sobel_y_2d = cv.filter2D(img_orig, cv.CV_32F, sobel_kernel_y_2d)

# gradient magnitude from 2D convolutions
magnitude_2d = np.sqrt(sobel_x_2d**2 + sobel_y_2d**2)
# Convert to uint8 for display
magnitude_2d_uint8 = cv.normalize(magnitude_2d, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

# 1D Sobel kernel for the x-direction
sobel_kernel_x_1d = np.array([[1, 0, -1]], dtype=np.float32)

# C1D Sobel kernel for the y-direction (transposed version of the x-direction kernel)
sobel_kernel_y_1d = np.array([[1], [2], [1]], dtype=np.float32)

# Convolve grayscale image with 1D kernel for y-direction
intermediate_y = cv.filter2D(img_orig, cv.CV_32F, sobel_kernel_y_1d)

# Convolve the result with 1D kernel for the x-direction
sobel_x_1d = cv.filter2D(intermediate_y, cv.CV_32F, sobel_kernel_x_1d)

# Convolve grayscale image with 1D kernel for x-direction
intermediate_x = cv.filter2D(img_orig, cv.CV_32F, sobel_kernel_x_1d)

# Convolve result with 1D kernel for y-direction
sobel_y_1d = cv.filter2D(intermediate_x, cv.CV_32F, sobel_kernel_y_1d)

# gradient magnitude from 1D convolutions
magnitude_1d = np.sqrt(sobel_x_1d**2 + sobel_y_1d**2)
# Convert to uint8 for display
magnitude_1d_uint8 = cv.normalize(magnitude_1d, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Display original image
ax[0].imshow(img_orig, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# Display gradient magnitude from 2D convolution
ax[1].imshow(magnitude_2d_uint8, cmap='gray')
ax[1].set_title('Sobel Magnitude (2D Convolution)')
ax[1].axis('off')

# Display gradient magnitude from 1D convolution
ax[2].imshow(magnitude_1d_uint8, cmap='gray')
ax[2].set_title('Sobel Magnitude (1D Convolution)')
ax[2].axis('off')

plt.tight_layout()
plt.show()