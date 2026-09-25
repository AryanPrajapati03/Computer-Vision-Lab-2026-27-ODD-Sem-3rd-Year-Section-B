import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load Image
# --------------------------------------------------

image = cv2.imread("image.jpg")

if image is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# OpenCV uses BGR format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display using OpenCV
cv2.imshow("Original Image - OpenCV", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display using Matplotlib
plt.figure(figsize=(6, 5))
plt.imshow(image_rgb)
plt.title("Original Image - Matplotlib")
plt.axis("off")
plt.show()


# --------------------------------------------------
# 2. Image Properties
# --------------------------------------------------

height, width, channels = image.shape

print("IMAGE PROPERTIES")
print("----------------")
print("Width:", width, "pixels")
print("Height:", height, "pixels")
print("Resolution:", width, "x", height)
print("Number of Channels:", channels)
print("Data Type:", image.dtype)
print("Shape:", image.shape)
print("Total Pixels:", width * height)
print("Minimum Pixel Value:", image.min())
print("Maximum Pixel Value:", image.max())


# --------------------------------------------------
# 3. Save Image in JPEG and PNG
# --------------------------------------------------

cv2.imwrite(
    "output_image.jpg",
    image,
    [cv2.IMWRITE_JPEG_QUALITY, 95]
)

cv2.imwrite("output_image.png", image)

print("\nImages saved as JPEG and PNG.")


# --------------------------------------------------
# 4. Color Space Conversion
# --------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


# --------------------------------------------------
# 5. Geometric Transformations
# --------------------------------------------------

# Resize
resized = cv2.resize(image, (400, 300))

# Rotation
center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    45,
    1.0
)

rotated = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)

# Horizontal Flip
horizontal_flip = cv2.flip(image, 1)

# Vertical Flip
vertical_flip = cv2.flip(image, 0)


# --------------------------------------------------
# 6. Negative / Complement Image
# --------------------------------------------------

negative = 255 - image


# --------------------------------------------------
# 7. Region of Interest (ROI)
# --------------------------------------------------

x1, y1 = 100, 100
x2 = min(400, width)
y2 = min(300, height)

roi = image[y1:y2, x1:x2]

print("\nROI PROPERTIES")
print("--------------")
print("ROI Shape:", roi.shape)
print("ROI Width:", roi.shape[1])
print("ROI Height:", roi.shape[0])
print("ROI Data Type:", roi.dtype)
print("ROI Minimum:", roi.min())
print("ROI Maximum:", roi.max())
print("ROI Mean:", roi.mean())


# --------------------------------------------------
# 8. Display All Processed Images
# --------------------------------------------------

plt.figure(figsize=(15, 12))

plt.subplot(3, 4, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(3, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(3, 4, 3)
plt.imshow(hsv)
plt.title("HSV")
plt.axis("off")

plt.subplot(3, 4, 4)
plt.imshow(lab)
plt.title("LAB")
plt.axis("off")

plt.subplot(3, 4, 5)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized")
plt.axis("off")

plt.subplot(3, 4, 6)
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title("Rotated 45 Degrees")
plt.axis("off")

plt.subplot(3, 4, 7)
plt.imshow(cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB))
plt.title("Horizontal Flip")
plt.axis("off")

plt.subplot(3, 4, 8)
plt.imshow(cv2.cvtColor(vertical_flip, cv2.COLOR_BGR2RGB))
plt.title("Vertical Flip")
plt.axis("off")

plt.subplot(3, 4, 9)
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative")
plt.axis("off")

plt.subplot(3, 4, 10)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("ROI")
plt.axis("off")

plt.tight_layout()
plt.show()