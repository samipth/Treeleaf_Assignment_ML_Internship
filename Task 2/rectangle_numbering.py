# Importing the required libraries
import cv2
import numpy as np


# Reading the image with cv2 module and assigning it ito variable.Here, 0 indicates images should be loaded as gray scale.
img = cv2.imread(r'C:\Users\samip\OneDrive\Desktop\image.png', 0)

# Convert the gray scale image to BGR format
output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Inverse threshold to convert image in binary format
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Finding contours in binary images
contours, [hist] = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# A list to store information about the rectangles in the image.
rect_list = []


for rect, (nxt, prv, first_child, parent) in zip(contours, hist):

    if parent == 0:
        _, _, line_index, _ = hist[first_child]
        line = contours[line_index]
        _, (width, height), _ = cv2.minAreaRect(line)
        line_length = max(width, height)
        rectangle = cv2.minAreaRect(rect)
        box = cv2.boxPoints(rectangle)
        box = np.intp(box)
        box_center = rectangle[0]

        rect_list.append((line_length, box, box_center))

# Sorting the rectangles by line length
rect_list.sort()

# Assigning the numbers based on length
num_rectangles = len(rect_list)
for index, (_, box, (x, y)) in enumerate(rect_list):
    if index < num_rectangles * 0.25:
        number = 1
    elif index < num_rectangles * 0.5:
        number = 2
    elif index < num_rectangles * 0.75:
        number = 3
    else:
        number = 4


    cv2.drawContours(output_img, [box], 0, (0, 255, 127), 3)
    
    # Decrease the font size and shift the text down by adding an offset to 'y' coordinate
    font_size = 1  # Decrease this value for smaller text
    text_offset_y = 20  # Increase this value to move the text further down
    cv2.putText(output_img, str(number), (round(x), round(y) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)
    
# displaying the output image
cv2.imshow("Image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()