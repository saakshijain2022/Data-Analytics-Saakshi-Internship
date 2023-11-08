# Data-Analytics-Saakshi-Internship

# classification task on the Iris dataset using a Decision Tree classifier

# Data Exploration and Visualization:
You loaded the Iris dataset into a DataFrame and checked its structure.
Explored the dataset for any missing values (no missing values found).
Visualized the data using various plots and a pairplot to gain insights into the relationships between the features and different species of iris flowers.
Calculated and visualized the correlation matrix between the numerical features.

# Data Preprocessing:
Removed the 'Id' column as it is not needed for the classification.
Split the dataset into features (x) and the target variable (y).
Split the dataset into training and testing sets using train_test_split.
Applied feature scaling to standardize the feature values.

# Model Building:
Built a Decision Tree classifier with the criterion set to 'entropy'.

# Model Evaluation:
Predicted the target values on the test dataset.
Evaluated the model's performance using a confusion matrix.
Calculated the accuracy score, which showed that the Decision Tree model achieved an accuracy of approximately 86.67%.
The confusion matrix indicates the number of correct and incorrect predictions for each class (Iris-setosa, Iris-versicolor, and Iris-virginica).


## Pencil Sketch Generator
This Python script uses the OpenCV library to convert a given image into a pencil sketch. 
It performs a series of image processing steps to achieve this effect, including grayscale conversion, inversion, Gaussian blur, and blending.

# Prerequisites
Before running the script, ensure you have OpenCV and NumPy installed. You can install OpenCV using pip:

pip install opencv-python

# Script Explanation
cv2.imread("your_image.jpg"): Loads the input image.

cv2.cvtColor(img, cv2.COLOR_BGR2GRAY): Converts the image to grayscale.

255 - gray_image: Inverts the grayscale image.

cv2.GaussianBlur(inverted_gray_image, (21, 21), 0): Applies Gaussian blur to the inverted image.

255 - blurred_image: Inverts the blurred image.

cv2.divide(gray_image, inverted_blurred_image, scale=256.0): Divides the grayscale image by the inverted blurred image to create the pencil sketch.

cv2.imshow(...): Displays the original image, grayscale image, and pencil sketch image.

cv2.waitKey(0): Waits for a key press to close the image windows.

## Python code that works with the Iris dataset, performs various machine learning tasks such as logistic regression, linear regression, k-nearest neighbors (KNN), and Naive Bayes classification, and includes data visualization using libraries like Matplotlib and Seaborn.
