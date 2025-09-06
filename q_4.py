import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_orig = cv.imread('a1images/spider.png')

# Convert the image BGR color space to HSV
img_hsv = cv.cvtColor(img_orig, cv.COLOR_BGR2HSV)

# # Split into H, S, V planes
h, s, v = cv.split(img_hsv)

# Define parameters for the transformation function
a = 0.55
sigma = 70

# Create an array representing possible intensity values (0 to 255).
x = np.arange(256)
t = np.minimum(x + a * 128 * np.exp(-((x - 128)**2) / (2 * sigma **2)), 255).astype('uint8')

# Apply transformation to S plane
s_transformed = cv.LUT(s, t)

# Merge the transformed Saturation channel back with the original channels.
img_transformed = cv.merge((h, s_transformed, v))
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
ax[0].set_title('Original Image')

# Display the vibrance-enhanced image in the second subplot.
ax[1].imshow(cv.cvtColor(img_transformed, cv.COLOR_HSV2RGB))
ax[1].set_title('Vibrance-enhanced Image')
for i in ax[0:2]:
    i.axis('off')

# Plot the intensity transformation function 't' in third subplot.
ax[2].plot(t)
ax[2].set_ylim([0, 255])
ax[2].set_xlim([0, 255])
ax[2].set_title(r'Intensity Transformation (a={a})'.format(a=a)) # Set the title, showing the 'a' value

plt.tight_layout()
plt.show()