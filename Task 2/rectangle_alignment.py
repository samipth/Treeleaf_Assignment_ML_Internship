import cv2
import numpy as np

def align_rectangles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    aligned_images = []

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width, height = rect[1]
        angle = rect[2] if width > height else rect[2] + 90

        rotation_matrix = cv2.getRotationMatrix2D(rect[0], angle, 1)
        aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        rect_points = cv2.transform(np.array([box]), rotation_matrix).squeeze().astype(int)
        x, y, w, h = cv2.boundingRect(rect_points)
        aligned_image = aligned_image[y:y + 2 * h, x:x + 2 * w]

        aligned_images.append(aligned_image)

    return aligned_images

# reading images
image = cv2.imread(r'C:\Users\samip\OneDrive\Desktop\image.png')
aligned_images = align_rectangles(image)

# displaying the aligned images
for i, aligned_image in enumerate(aligned_images):
    cv2.imshow(f'Aligned Image {i + 1}', aligned_image)

# closing displayed windows
cv2.waitKey(0)
cv2.destroyAllWindows()
