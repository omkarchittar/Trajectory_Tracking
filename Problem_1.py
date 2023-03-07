import cv2
import numpy as np
import cmath
import csv
import matplotlib.pyplot as plt

def centroid(image):
  y_values, x_values = np.nonzero(image)
  if x_values.size == 0:
    X = np.nan
  else:
    zero_count = np.count_nonzero(x_values == 0)
    nan_count = np.count_nonzero(np.isnan(x_values))
    inf_count = np.count_nonzero(np.isinf(x_values))
    if zero_count > 0 or nan_count > 0 or inf_count > 0:
        X = np.nan
    else:
        X = np.sum(x_values)/x_values.shape[0]

  if y_values.size == 0:
    Y = np.nan
  else:
    zero_count = np.count_nonzero(y_values == 0)
    nan_count = np.count_nonzero(np.isnan(y_values))
    inf_count = np.count_nonzero(np.isinf(y_values))
    if zero_count > 0 or nan_count > 0 or inf_count > 0:
        Y = np.nan
    else:
        Y = np.sum(y_values)/y_values.shape[0]

  return (int(X), int(Y))

video = cv2.VideoCapture('ball.mov')
image = cv2.imread('frame.jpg')

lower_red = np.array([5, 6, 130], dtype = "uint8") 
upper_red= np.array([60, 60, 255], dtype = "uint8")
filtered = cv2.inRange(image, lower_red, upper_red)
output = cv2.bitwise_and(image, image, mask = filtered)

i=0
frames = []
success, frame = video.read()
success = True
while success:
    success, frame = video.read()
    if success == True:
      frames.append(frame) 

frame_copy = frames[1]
gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_image,0,50,0)

print("Number of frames captured: ", len(frames), "\n")

#Mark centroid on ball
try:
  cX, cY = centroid(thresh)
  cv2.circle(thresh, (cX, cY), 5, (255, 255, 255), -1)
  cv2.putText(thresh, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
except:
  None

cv2.imshow("",frame_copy)
cv2.imshow("",thresh)

#Extract x,y coordinate of ball from each frame of video
coordinates = []
for images in frames[:]:
    filtered = cv2.inRange(images, lower_red, upper_red)
    output = cv2.bitwise_and(images, images, mask = filtered) 
    gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,0,255,0)

    try:
      cX, cY = centroid(thresh)
      cv2.circle(images, (cX, cY), 5, (255, 255, 255), -1)
      specific_coordinate = (cX, cY)
      coordinates.append(specific_coordinate)
    except:
      continue

im = plt.imread('frame.jpg')
implot = plt.imshow(im)

#x and y coordinates of all the detected ball in frames
x = [i[0] for i in coordinates]
y = [i[1] for i in coordinates]

x_data = np.array(x)
y_data = np.array(y)

#Scatter plot for visualization of selected coordinates from video
plt.scatter(x_data, y_data, s=10, marker='o', color="red")
plt.show()

def curve_fit(x_data, y_data, degree=2):
    n = len(x_data)

    X = np.ones((n, degree + 1))
    for j in range(degree):
        X[:, j] = x_data ** (degree - j)

    # Solve the least-squares problem
    XT_X = np.dot(X.T, X)
    XT_Y = np.dot(X.T, y_data)
    coeffs = np.linalg.solve(XT_X, XT_Y)

    # Generate new x-values for fitting curve
    x_fit = np.linspace(x_data.min(), x_data.max(), 1000)

    # Calculate corresponding y-values
    y_fit = np.zeros_like(x_fit)
    for j in range(degree + 1):
        y_fit += coeffs[j] * x_fit ** (degree - j)

    return x_fit, y_fit, coeffs

x_fit, y_fit, coeffs = curve_fit(x_data, y_data)

[a, b, c] = coeffs

im = plt.imread('frame.jpg')
implot = plt.imshow(im)
plt.plot(x_data, y_data, '.', label='Data', color = 'red')
plt.plot(x_fit, y_fit, label='Fitted Curve', color = 'blue')
plt.legend()
plt.show()

print('Equation of the fitted curve is: ')
print('y = %f + (%f)x + (%f)x^2' %(c,b,a), "\n")

# At the initial frame, x1 = 0,
y1 = a*(0) + b*(0) + c

# The ball lands 300 pixels below when first detected,
y2 = y1 + 300

# Therefore, 
c = y1 - y2

# calculating the discriminant
dis = (b**2) - (4 * a*c)
  
# find two results
x1 = (-b - cmath.sqrt(dis))/(2 * a)
x2 = (-b + cmath.sqrt(dis))/(2 * a)
  
# printing the results
print('The roots are')
print(x1)
print(x2, "\n")

print("X co-ordinate of the landing spot = ", x2)