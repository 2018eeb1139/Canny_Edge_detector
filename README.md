
# Canny Edge Detection

The Canny edge detector is an edge detection operator that detects a wide range of edges in images using a multi-stage approach. It's an image processing technique for detecting edges in a picture while reducing noise. It helps in extracting useful structural information from different vision objects, reducing the amount of data to be processed dramatically. It's been used in a variety of computer vision systems. Canny discovered that the criteria for using edge detection on a variety of vision systems are remarkably similar.

# Process in Canny Edge Detection

* Noise reduction using Gaussian filter 
 
* Gradient calculation along the horizontal and vertical axis 
 
* Non-Maximum suppression of false edges 
 
* Double thresholding for segregating strong and weak edges 
 
* Edge tracking by hysteresis.

# Explanation of code

* In main function called the Canny_detector function.
* In this function we first convert the image to gray scale. 
* Then to reduce noise we use gaussian filter.
* Then we calculated the gradient using sobel operator.
* To find false edges we called Non_maximum_suppression function.
* After this function calling we make another function call which is Double_thresholding function. And did edge tracking by hystersis. 
* Used the inbuilt SSIM code for similarity index calculation.


To read the image2 and image3 we have to uncomment the desired line(84,85) in the code.

# Hi, I'm Aman Chandra! 👋


## 🔗 Links

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aman-chandra-51993b16b/)



## 🛠 Skills
Javascript, HTML, CSS...

