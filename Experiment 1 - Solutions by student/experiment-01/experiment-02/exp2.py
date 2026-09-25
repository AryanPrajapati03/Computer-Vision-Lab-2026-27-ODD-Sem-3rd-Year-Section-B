# Image Enhancement using Histogram Techniques
# Contrast Stretching, Histogram Equalization, and CLAHE

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Step 1 & 2: Load image and convert to grayscale
# ---------------------------------------------------------

# Change this to your image filename
image_path = "low_contrast.jpg"

image = cv2.imread("image.jpg")

if image is None:
    print("Error: Could not load the image.")
    print("Make sure the image exists in the same folder as this Python file.")
    exit()

# Convert BGR image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------
# Step 3: Display original image and its histogram
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.hist(gray.ravel(), bins=256, range=[0, 256], color="black")
plt.title("Histogram - Original Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Step 4: Contrast Stretching
# ---------------------------------------------------------

# Find minimum and maximum pixel values
min_val = np.min(gray)
max_val = np.max(gray)

# Contrast stretching formula:
# New Pixel = (Pixel - min) * 255 / (max - min)

if max_val != min_val:
    contrast_stretched = ((gray - min_val) * 255 /
                          (max_val - min_val)).astype(np.uint8)
else:
    contrast_stretched = gray.copy()


# ---------------------------------------------------------
# Step 5: Histogram Equalization
# ---------------------------------------------------------

hist_equalized = cv2.equalizeHist(gray)


# ---------------------------------------------------------
# Step 6: CLAHE
# ---------------------------------------------------------

# Create CLAHE object
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(gray)


# ---------------------------------------------------------
# Step 7: Display enhanced images and histograms
# ---------------------------------------------------------

plt.figure(figsize=(15, 10))

# Original
plt.subplot(3, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.hist(gray.ravel(), bins=256, range=[0, 256], color="black")
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")


# Contrast Stretching
plt.subplot(3, 2, 3)
plt.imshow(contrast_stretched, cmap="gray")
plt.title("Contrast Stretched")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.hist(
    contrast_stretched.ravel(),
    bins=256,
    range=[0, 256],
    color="blue"
)
plt.title("Contrast Stretched Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")


# Histogram Equalization
plt.subplot(3, 2, 5)
plt.imshow(hist_equalized, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(3, 2, 6)
plt.hist(
    hist_equalized.ravel(),
    bins=256,
    range=[0, 256],
    color="green"
)
plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Display CLAHE separately
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(clahe_image, cmap="gray")
plt.title("CLAHE Enhanced Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.hist(
    clahe_image.ravel(),
    bins=256,
    range=[0, 256],
    color="red"
)
plt.title("CLAHE Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Step 8 & 9: Visual comparison
# ---------------------------------------------------------

plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(contrast_stretched, cmap="gray")
plt.title("Contrast Stretching")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(hist_equalized, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(clahe_image, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Step 10: Save enhanced images
# ---------------------------------------------------------

cv2.imwrite("contrast_stretched.jpg", contrast_stretched)
cv2.imwrite("histogram_equalized.jpg", hist_equalized)
cv2.imwrite("clahe_enhanced.jpg", clahe_image)

print("Image enhancement completed successfully!")
print("Enhanced images have been saved.")