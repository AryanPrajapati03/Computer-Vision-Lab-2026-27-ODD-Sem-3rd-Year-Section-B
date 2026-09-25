# ============================================================
# Experiment No. 7
# Motion Estimation using Optical Flow Algorithms
# ============================================================

# Required Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


# ============================================================
# 1. LOAD VIDEO
# ============================================================

# Give the path of your video file here
video_path = "input video.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)

# Check whether video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    print("Check the video path.")
    exit()

# Read the first frame
ret, first_frame = cap.read()

if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Resize frame if required
scale = 0.75
first_frame = cv2.resize(
    first_frame,
    None,
    fx=scale,
    fy=scale
)

# Convert first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)


# ============================================================
# 2. LUCAS-KANADE SPARSE OPTICAL FLOW
# ============================================================

# Parameters for Shi-Tomasi corner detection
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

# Detect feature points in the first frame
prev_points = cv2.goodFeaturesToTrack(
    prev_gray,
    mask=None,
    **feature_params
)

# Create mask for drawing trajectories
lk_mask = np.zeros_like(first_frame)

# Generate random colors for feature points
colors = np.random.randint(
    0, 255,
    (100, 3)
)


# ============================================================
# 3. FARNEBACK DENSE OPTICAL FLOW PARAMETERS
# ============================================================

# Parameters for Farneback algorithm
farneback_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)


# ============================================================
# 4. VIDEO PROCESSING
# ============================================================

frame_count = 0

lk_times = []
farneback_times = []

while True:

    # Read next frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale
    )

    # Convert frame to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # LUCAS-KANADE OPTICAL FLOW
    # ========================================================

    start_time = time.time()

    if prev_points is not None:

        # Calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points,
            None,
            **lk_params
        )

        # Select only successfully tracked points
        if next_points is not None:

            good_new = next_points[
                status == 1
            ]

            good_old = prev_points[
                status == 1
            ]

        else:
            good_new = []
            good_old = []

    else:
        good_new = []
        good_old = []

    lk_time = time.time() - start_time
    lk_times.append(lk_time)


    # ========================================================
    # DRAW LUCAS-KANADE MOTION VECTORS
    # ========================================================

    for i, (new, old) in enumerate(
        zip(good_new, good_old)
    ):

        # Convert coordinates to integers
        x_new, y_new = new.ravel()
        x_old, y_old = old.ravel()

        x_new = int(x_new)
        y_new = int(y_new)
        x_old = int(x_old)
        y_old = int(y_old)

        # Draw trajectory line
        lk_mask = cv2.line(
            lk_mask,
            (x_new, y_new),
            (x_old, y_old),
            colors[i].tolist(),
            2
        )

        # Draw current point
        frame = cv2.circle(
            frame,
            (x_new, y_new),
            5,
            colors[i].tolist(),
            -1
        )

        # Draw arrow showing direction
        frame = cv2.arrowedLine(
            frame,
            (x_old, y_old),
            (x_new, y_new),
            (0, 255, 0),
            2,
            tipLength=0.3
        )


    # ========================================================
    # FARNEBACK DENSE OPTICAL FLOW
    # ========================================================

    start_time = time.time()

    dense_flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        **farneback_params
    )

    farneback_time = time.time() - start_time
    farneback_times.append(farneback_time)


    # ========================================================
    # CONVERT DENSE FLOW TO COLOR VISUALIZATION
    # ========================================================

    # Calculate magnitude and angle
    magnitude, angle = cv2.cartToPolar(
        dense_flow[..., 0],
        dense_flow[..., 1]
    )

    # HSV image
    hsv = np.zeros_like(frame)

    # Saturation
    hsv[..., 1] = 255

    # Direction of motion
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Magnitude of motion
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Convert HSV to BGR
    dense_flow_color = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


    # ========================================================
    # COMBINE LUCAS-KANADE RESULT
    # WITH DENSE FLOW RESULT
    # ========================================================

    lk_result = cv2.add(
        frame,
        lk_mask
    )

    # Add labels
    cv2.putText(
        lk_result,
        "Lucas-Kanade Sparse Optical Flow",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        dense_flow_color,
        "Farneback Dense Optical Flow",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    # Resize for side-by-side display
    display_lk = cv2.resize(
        lk_result,
        (640, 480)
    )

    display_dense = cv2.resize(
        dense_flow_color,
        (640, 480)
    )

    # Combine both results
    combined = np.hstack(
        (display_lk, display_dense)
    )

    # Show result
    cv2.imshow(
        "Optical Flow Comparison",
        combined
    )


    # ========================================================
    # UPDATE PREVIOUS FRAME AND POINTS
    # ========================================================

    prev_gray = gray.copy()

    if len(good_new) > 0:

        # Convert points to correct format
        prev_points = np.array(
            good_new,
            dtype=np.float32
        ).reshape(-1, 1, 2)

    else:

        # Detect new features if points are lost
        prev_points = cv2.goodFeaturesToTrack(
            gray,
            mask=None,
            **feature_params
        )

        # Reset trajectory mask
        lk_mask = np.zeros_like(frame)


    frame_count += 1


    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


# ============================================================
# 5. RELEASE VIDEO
# ============================================================

cap.release()
cv2.destroyAllWindows()


# ============================================================
# 6. PERFORMANCE ANALYSIS
# ============================================================

if len(lk_times) > 0:

    average_lk_time = np.mean(lk_times)

else:

    average_lk_time = 0


if len(farneback_times) > 0:

    average_farneback_time = np.mean(
        farneback_times
    )

else:

    average_farneback_time = 0


print("\n==========================================")
print("       OPTICAL FLOW PERFORMANCE")
print("==========================================")

print("Total frames processed:",
      frame_count)

print(
    "Average Lucas-Kanade processing time:",
    round(average_lk_time * 1000, 3),
    "ms/frame"
)

print(
    "Average Farneback processing time:",
    round(farneback_time * 1000, 3),
    "ms/frame"
)

print("==========================================")
