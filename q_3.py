
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_orig = cv.imread('a1images/highlights_and_shadows.jpg')

# Convert to LAB color space
img_lab = cv.cvtColor(img_orig, cv.COLOR_BGR2LAB)
# Extract L channel
img_l = img_lab[:, :, 0].copy()

# Apply gamma correction to L channel
gamma = 0.7
t = np.array([(i/255)** gamma * 255 for i in np.arange(256)], dtype='uint8')
gamma_corrected_l = cv.LUT(img_l, t)

# Replace L channel with corrected values
img_lab[:, :, 0] = gamma_corrected_l

# Convert back to BGR color space
img_corrected = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)

# # Histogram of original and corrected L channel
hist_orig, bins_orig = np.histogram(img_l.ravel(), 256, [0, 256])
hist_corrected, bins_corrected = np.histogram(gamma_corrected_l.ravel(), 256, [0, 256])

fig, axis = plt.subplots(2, 2, figsize=(12, 8))
axis[0,0].imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
axis[0,0].set_title('Original Image')
axis[0,0].axis('off')
axis[0,1].imshow(cv.cvtColor(img_corrected, cv.COLOR_BGR2RGB))
axis[0,1].set_title(r'Corrected Image ($\gamma = {gamma}$)'.format(gamma=gamma))
axis[0,1].axis('off')

axis[1,0].plot(hist_orig)
axis[1,0].set_title('Histogram of L Plane- Original')
axis[1,0].set_xlabel('Intensity Value')
axis[1,0].set_ylabel('Frequency')
axis[1,1].plot(hist_corrected)
axis[1,1].set_title('Histogram of L Plane- Corrected')
axis[1,1].set_xlabel('Intensity Value')
axis[1,1].set_ylabel('Frequency')

plt.show()