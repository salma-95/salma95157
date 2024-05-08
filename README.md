## BOX
```
 os: Operating System dependent functionality (for file operations).

csv: Reading and writing CSV files.

 PIL.Image and PIL.ImageDraw: Working with images and drawing on them.


import os
import csv
from PIL import Image, ImageDraw

 csv_file: Path to the CSV file containing bounding box data.

 image_dir: Directory containing the images.

  output_dir: Directory where the output images with bounding boxes will be saved.



csv_file = "/home/salma-mohammad/Downloads/7622202030987_bounding_box.csv"
image_dir = "/home/salma-mohammad/Downloads/7622202030987"
output_dir = "/home/salma-mohammad/Downloads/7622202030987_with_boxes"
``
draw_boxes(image, boxes): Draws bounding boxes on the given image.

crop_image(image, boxes): Crops the image based on the bounding boxes

os.makedirs(output_dir, exist_ok=True)


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        draw.rectangle([left, top, right, bottom], outline="red")
    return image


def crop_image(image, boxes):
    cropped_images = []
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        cropped_img = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)
    return cropped_images

The script opens the CSV file and iterates over each row.
For each row, it gets the filename of the corresponding image, opens the image, and reads the bounding box coordinates from the CSV row.
It then draws bounding boxes on the image and saves it with prefixed 'full_' in the output directory.
It also crops the image based on the bounding box coordinates and saves each cropped image with prefixed index and the original filename in the output directory


with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        cropped_images = crop_image(image, boxes)
        for i, cropped_img in enumerate(cropped_images):
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))
```
Images are saved using Image.save() method from the PIL library.

## Imagehistogram
```
    numpy as np: For numerical operations.

    cv2 as cv: OpenCV library for image processing.

    matplotlib.pyplot as plt: For plotting histograms.

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cv.imread('/home/salma-mohammad/rose.jpg'): Reads the image named 'rose.jpg' from the specified path.

cv.imwrite("/home/salma-mohammad/myexp/lotus", img): Saves the image in the specified path ('/home/salma-mohammad/myexp/lotus'). Note that the file extension (e.g., '.jpg', '.png') should be included in the filename.

 assert img is not None, "file could not be read, check with os.path.exists()": Checks if the image was successfully read. If the image is not read, it raises an AssertionError with the given message.
    The script calculates histograms for each color channel (Blue, Green, Red) using cv.calcHist() function.

    It iterates over each color channel and calculates the histogram using the pixel values in that channel.

    Histograms are plotted using plt.plot() function from matplotlib.

    The plt.xlim([0, 256]) sets the limits of the x-axis to be from 0 to 256, which are the possible pixel intensity values.

    Finally, plt.show() displays the plotted histograms.

img = cv.imread('/home/salma-mohammad/rose.jpg')
cv.imwrite("/home/salma-mohammad/myexp/lotus",img)
assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()



## iteration
 num = list(range(10)): Creates a list containing numbers from 0 to 9.

  previousNum = 0: Initializes the variable previousNum to 0.

    The loop iterates over each element i in the num list.

   Within each iteration:

 It calculates the sum of the current number i and the previous number previousNum.


 It prints the current number i, the previous number previousNum, and their sum.

 Inside the loop, previousNum is updated incorrectly. It should be updated to the current value of i after calculating the sum, but it's assigned i directly.

 Because of this, previousNum always holds the same value as i, and the sum is not calculated correctly.

To fix the issue, update previousNum to the sum of the current number i and the previous number previousNum before moving to the next iteration.

  num = list(range(10))
  previousNum = 0
   for i in num:
    sum = previousNum + i
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum)) # <- This is the issue.
    previousNum=i
     ```


## video
```
import cv2: This imports the OpenCV library, which is used for video capture and processing

vid = cv2.VideoCapture(0): This initializes a video capture object (vid) for the default camera (index 0). You can also specify the index of the desired camera if you have multiple cameras connected.

while(True):: This initiates an infinite loop for continuously capturing frames from the video stream.

ret, frame = vid.read(): This captures a frame from the video stream and stores it in the variable frame. The return value ret indicates whether the frame was successfully captured.

cv2.imshow('frame', frame): This displays the captured frame in a window with the title 'frame'. The imshow() function takes the window title and the image to be displayed as arguments

if cv2.waitKey(1) & 0xFF == ord('q'): break: This statement checks for the key press event. If the key pressed is 'q' (which corresponds to the ASCII value 113), it breaks out of the loop and terminates the program.

vid.release(): This releases the video capture object once the loop is exited. It's essential to release the camera resource after usage.



# import the opencv library 
import cv2 
  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()
```

