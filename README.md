
## Scanner.py 
### Document/receipt scanner 
---
### First cell
In first cell I imported the necessary libraries, the test image and I did some modifications on it, such as : resizing (ratio) , grayscaling, adding gaussian blur, finding edges with canny function. 



```python
#FIRST CELL

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import heapq as hq

# Load image from project dir
image = cv.imread("4.jpg")

# Copy the original image
original = image.copy()

# Resize the image by height = 500px ( bigger sizes awful for OpenCV)
height = 500
ratio = height / image.shape[0]
image = cv.resize(image,(int(ratio * image.shape[1]), height))
resized = image.copy()
pointed = image.copy()

# Grayscale image
grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = grayscale.copy()
# Add gaussian blur to help detect edges (not to smooth)
# https://docs.opencv.org/3.4.7/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
blur = cv.GaussianBlur(grayscale, (7, 7), 0)
blurred = blur.copy()

# Finding the egdes with Canny function
# https://docs.opencv.org/3.4.7/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
# https://docs.opencv.org/3.4.7/da/d22/tutorial_py_canny.html
# image, thresolding values: minvalue:,maxvalue:
edge = cv.Canny(blur, 75, 250)

```

### Second, third cell.........

#### Method 1:
In second cell I found the edges, and I sorted them to show only the edges.


```python
#SECOND CELL

# Conture
#https://docs.opencv.org/3.4.7/d4/d73/tutorial_py_contours_begin.html
#https://docs.opencv.org/3.4.7/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
#https://docs.opencv.org/3.4.7/dd/d49/tutorial_py_contour_features.html
#https://docs.opencv.org/3.4.7/d3/dc0/group__imgproc__shape.html#ga2c759ed9f497d4a618048a2f56dc97f1
#https://docs.opencv.org/3.4.7/d3/dc0/group__imgproc__shape.html#ga8d26483c636be6b35c3ec6335798a47c
#https://docs.opencv.org/3.4.7/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c

# first argument: output of the canny edge detection
# retrieval method: RETR_LIST : retrieves all of the contours without establishing any hierarchical relationships.
# contour approximation method: CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
contours = cv.findContours(edge.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
contours = contours[1]
# Extracting boundaries with sorting in the reverse order to find the biggest contour. 
contours = sorted(contours, key=cv.contourArea, reverse=True)
for c in contours:
    #Using arcLength which tries to find a square(perimeter) in the contours
    perimeter = cv.arcLength(c, True) # true for closed contour
    approx = cv.approxPolyDP(c,0.02*perimeter,True) #first value contours, second epsylon value for: Parameter specifying the approximation accuracy. This is the maximum distance between the original curve and its approximation.
    # If approx is 4 then it is a square
    if len(approx) == 4:
        square = approx
        break

#https://docs.opencv.org/3.3.1/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc        
squared = cv.drawContours(image, [square], -1, (0, 255, 0), 5)

```


```python
##### THIRD CELL

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html?highlight=argmin#numpy.argmin
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html?highlight=argmax#numpy.argmax
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.diff.html

#Getting the 4 points ordered


# Reshaping the square array "outer"
points = square.reshape(4, 2)

#print("Points not ordered\n",pts)

# Creating a numpy array with zeros
rect = np.zeros((4, 2), dtype = "float32")

summas = points.sum(axis = 1)

rect[0] = points[np.argmin(summas)]  #Top-left
rect[2] = points[np.argmax(summas)]  #Bottom-right

diff = np.diff(pts, axis = 1)
rect[1] = points[np.argmin(diff)]   #Top-right
rect[3] = points[np.argmax(diff)]   #Bottom-left


#print("\nPoints ordered clockwise\n",rect)

#For testing purposes
for i,j in rect:
    cv.circle(pointed,(i,j), 7, (0,255,0), -1)
```


```python
#FOURTH CELL
#Calculate points for original image size
rect = rect/ratio

#Getting each x y value from rect 
(tl, tr, br, bl) = rect

#Calculate width
widthTop = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
widthBottom = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
maxWidth = max(int(widthTop),int(widthBottom))

#Calculate height
heightLeft = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
heightRight = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
maxHeight = max(int(heightLeft),int(heightRight))


#Creating the new size of the image in array
output = np.array([
    [0,0],
    [maxWidth,0],
    [maxWidth,maxHeight],
    [0,maxHeight]], dtype = "float32")


# Applying transformation and warping
transform = cv.getPerspectiveTransform(rect, output)
warped = cv.warpPerspective(original, transform, (maxWidth, maxHeight))

bw = cv.cvtColor(warped, cv.COLOR_RGB2GRAY)

ret, thr = cv.threshold(bw,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

```


```python
cv.imwrite("1d_resized.jpg", resized)
cv.imwrite("2d_bw.jpg",gray)
cv.imwrite("3d_blur.jpg",blurred)
cv.imwrite("4d_cannyedged.jpg",edge)
cv.imwrite("5d_squared.jpg",squared)
cv.imwrite("6d_pointed.jpg",pointed)
cv.imwrite("7d_warped.jpg",warped)
cv.imwrite("8d_final.jpg",thr)

#cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 3
#threshold(src, -1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

#### Method 2:
In second cell I found the edges, and I sorted them to show only the edges.


```python
a = np.array(edge)
b = {} # horizontal
e = []
c = [] # vertical

height, width = image.shape[:2]
print(height)
print(width)

for l in range(0, width):
    b[np.mean(a[:, l])] = l

for i in range(0, width):
    e.append(np.mean(a[:, i]))

    
for j in range(0, height):
    c.append(np.mean(a[j,:]))


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(e)
plt.subplot(122)
plt.plot(c)
plt.show()

#print(max(b))
#for asd in b:
    #print(asd,b[asd])
    
#key, value = max(b.iteritems(), key=lambda x:x[1])
#print(max(b.values()))   

#print(max(b.keys()),max(b.values()))
```
# Original image
(4.jpg)
(1d_resized.jpg)
cv.imwrite("2d_bw.jpg",gray)
cv.imwrite("3d_blur.jpg",blurred)
cv.imwrite("4d_cannyedged.jpg",edge)
cv.imwrite("5d_squared.jpg",squared)
cv.imwrite("6d_pointed.jpg",pointed)
cv.imwrite("7d_warped.jpg",warped)
cv.imwrite("8d_final.jpg",thr)
