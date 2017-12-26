#### Pipeline
This part covers detail sequence of steps that will help finding out the lanes.

- Generate gray scale image
- Apply gaussian blur to gray scale image to smoothen edges
- Apply canny edge detection on gaussian blur smoothened image
- Identify the region of interest
- Apply Hough transformation and identify the coordinates of the lines
- Separate out left and right lines based on the slope. Left lane is of negative slope and right lane is positive
- Add the coordinates of negative slope to left lane line coordinates and coordinates of positive slope to right lane line coordinates
- Ignore coordinates with large slopes.
- Ployfit all the line to generate an optimal line for both left and right lane
- Use the slopes of the optimal lines and generate line to fit in the region of interest
- Add the lines to a queue holding last 20 average lines generated.
- This averaging will help smoothen the lanes.

Much of the above optimizations that I have implemented are from the blogs suggested in udacity discussion forum.

#### Shortcomings
- This pipeline does not work on curves
- This pipeline has a specific region of interest.
- Challenge video from 5 seconds to 7 seconds the color of the road becomes gray and the lanes could not be identified.

#### Improvements
Graying image is always not useful. Multiple image filters have to be applied and averaged to find the optimal way to identify the lanes in different color roads and disturbances like shadows of trees etc. I might not have correctly applied the averaging of the code here. Need to identify an optimal algorithm.

