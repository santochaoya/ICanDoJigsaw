import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def extract_shape():
    # load images
    img_dir = r'data/'
    img1 = cv2.imread(os.path.join(img_dir, '1.jpeg'))

    # Convert to grey scale
    imgGray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(imgGray, threshold1=30, threshold2=100)

    # show images
    plt.imshow(edges)
    plt.show()

def extract_shape_by_color():
    
    # load images
    img_dir = r'data/'
    img = cv2.imread(os.path.join(img_dir, '1.jpeg'))

    # puts 0 to the white (background) and 255 in other places (greyscale value < 250)
    _, thresholded = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)

    # gets the labels and the amount of labels, label 0 is the background
    amount, labels = cv2.connectedComponents(thresholded)

    # lets draw it for visualization purposes
    preview = np.zeros((img.shape[0], img.shape[2], 3), dtype=np.uint8)

    print (amount) #should be 3 -> two components + background

    # draw label 1 blue and label 2 green
    preview[labels == 1] = (255, 0, 0)
    preview[labels == 2] = (0, 255, 0)

    cv2.imshow("frame", preview)
    cv2.waitKey(0)


if __name__ == '__main__':
    extract_shape_by_color()