# Image-Feature-Extraction

Image Feature Extraction Code
This repository contains a set of image feature extraction methods using Python libraries such as OpenCV, scikit-image, and SciPy. These methods compute various image statistics and properties useful for image analysis, including Local Binary Pattern (LBP), Histogram of Oriented Gradients (HOG), edge detection, Gabor filtering, and more. The features computed can be used for applications like facial recognition, image classification, and other computer vision tasks.

Libraries Used:
NumPy: For numerical operations.
OpenCV: For image processing and computer vision tasks.
scikit-image: For image feature extraction (e.g., LBP, HOG).
SciPy: For statistical and signal processing.
pywt (PyWavelets): For wavelet transformations.
Functions Overview:
1. a1(i, p=8, r=1)
Computes the combined LBP descriptor, including mean, variance, skewness, and kurtosis of the LBP histogram.

Parameters:
i: Input image (grayscale).
p: Number of LBP points (default is 8).
r: Radius of the LBP neighborhood (default is 1).
Returns: Mean, variance, skewness, and kurtosis of the LBP histogram.
2. a2(i, p=8, r=1)
Computes the uniformity score of the LBP histogram.

Parameters: Same as a1.
Returns: Uniformity score (energy of the LBP histogram).
3. a3(i, p=8, r=1, n=3)
Finds the top n histogram peaks from the LBP descriptor.

Parameters: Same as a1 and a2.
Returns: Sorted peaks of the LBP histogram.
4. a4(i)
Computes the mean and variance of the color channels (Red, Green, Blue) of an image.

Parameters: Input image in BGR format.
Returns: Mean and variance for each color channel (Red, Green, Blue).
5. a5(i)
Computes the mean and variance of the grayscale histogram.

Parameters: Input image (grayscale).
Returns: Mean and variance of the grayscale histogram.
6. a6(i)
Computes the mean and variance of the Histogram of Oriented Gradients (HOG) features.

Parameters: Input image.
Returns: Mean and variance of the HOG features.
7. a7(i, o=9, p=(8, 8), c=(2, 2))
Counts the number of significant peaks in the HOG histogram.

Parameters: o: Number of orientations (default is 9), p: Pixels per cell, c: Cells per block.
Returns: Number of significant peaks in the HOG feature vector.
8. a8(i)
Computes the edge density using the Sobel filter (gradient magnitude).

Parameters: Input image.
Returns: Mean gradient magnitude.
9. a9(i, l=100, h=200)
Computes the edge density using the Canny edge detection.

Parameters: l and h are the low and high thresholds for Canny.
Returns: Edge density as a fraction of edge pixels.
10. a10(i, f=[0.4, 0.5, 0.6], t=[0, np.pi/4, np.pi/2, 3*np.pi/4])
Computes the response variance of the Gabor filter at different frequencies and orientations.

Parameters: f and t represent the frequency and orientation of the Gabor filters.
Returns: List of Gabor filter response variances.
11. a11(i, f=[0.4, 0.5, 0.6], t=[0, np.pi/4, np.pi/2, 3*np.pi/4])
Computes the mean values of the Gabor filter response at different frequencies and orientations.

Parameters: Same as a10.
Returns: List of Gabor filter mean values.
12. a12(i, pth)
Detects faces in the image using OpenCV's Haar cascade classifier, and computes a thresholding sum based on the detected face.

Parameters: pth is the path to the Haar cascade XML file for face detection.
Returns: Sum of thresholded face region.
13. a13(i, p=8, r=1)
Computes the sum of squared LBP histogram features.

Parameters: Same as a1.
Returns: Sum of squared LBP histogram values.
14. a14(i)
Computes the mean gradient magnitude (Sobel filter).

Parameters: Input image.
Returns: Mean gradient magnitude.
15. a15(i, t=100)
Computes the mean and variance of the pixel values below a certain threshold.

Parameters: t is the threshold value for pixel intensity.
Returns: Mean and variance of the pixel values less than the threshold.
16. a16(i)
Computes the image difference between the left and right halves of the image.

Parameters: Input image.
Returns: Mean squared difference between the left and right halves of the image.
17. a17(i)
Computes the mean and variance of the approximation coefficients of a 2D Haar wavelet transform.

Parameters: Input image.
Returns: Mean and variance of the approximation coefficients.
18. a18(i)
Computes the mean and variance of the combined horizontal, vertical, and diagonal detail coefficients of a 2D Haar wavelet transform.

Parameters: Input image.
Returns: Mean and variance of the combined detail coefficients.
