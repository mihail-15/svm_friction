# svm_friction
svm_friction is a Python tool that performs data analysis, calculates, and predicts the coefficient of friction (COF) of open-cell AlSi10Mg-SiC composites, using support vector regression (SVR).

Installation

To install svm_friction, you need to have Python 3.9 or higher and the following packages: pandas, numpy, sklearn, matplotlib, and patheffects. 

You can choose either pip or conda commands to install the packages.

You can download or clone the GitHub repository at https://github.com/mihail-15/svm_friction. The repository contains the following files:

svm_friction.py: The main script that performs data analysis and modeling on the COF of open-cell AlSi10Mg-SiC composites using SVR.

B_1.xlsx, B_2.xlsx, B_3.xlsx: The input data files containing the experimental data for each dataset of the open-cell AlSi10Mg-SiC composites.

B_descriptive_statistics.txt: The output file containing the descriptive statistics of the average COF list.

Average_COF_B.xlsx: The output file containing the average COF list.

Average_time_B.xlsx: The output file containing the average time list.

COF_B.xlsx: The output file containing the merged data of average time and average COF.

B_performance_metrics.txt: The output file containing the performance metrics of the SVR model on the test and validation sets.

pred_COF_B.png: The output image file containing the scatter plot of the actual vs predicted COF as a function of time for material B.

hist_COF_B.png: The output image file containing the histogram of the COF for material B.

box_COF_B.png: The output image file containing the boxplot of the COF for material B.

README.md: This file.

Usage

To use svm_friction, you need to run the svm_friction.py script in a Python interpreter or an IDE such as Spyder or PyCharm. You need to make sure that the input data files (B_1.xlsx, B_2.xlsx, B_3.xlsx) are in the same folder as the script. The script will output the following files: B_descriptive_statistics.txt, Average_COF_B.xlsx, Average_time_B.xlsx, COF_B.xlsx, B_performance_metrics.txt, pred_COF_B.png, hist_COF_B.png, and box_COF_B.png.

The script performs the following steps:

Read each .xlsx file containing the data for each dataset of the open-cell AlSi10Mg-SiC composites (B1, B2, B3) using the pandas.read_excel() function.

Assign the dataset to a variable using the pandas.DataFrame() function. The dataset contains two columns: TIME (time in sec.) and FF (friction force in N).

Calculate the COF for each dataset using the formula: COF = friction force / normal force (100 N) using the pandas.Series() function.

Create an empty list to store the average COF and another empty list to store the average time using the list() function.

Loop through the COF and time values of each dataset using the range() and len() functions and append the average COF and time values to the corresponding lists using the list.append() function.

Calculate the mean, median, standard deviation, minimum, and maximum of the average COF list using the numpy.mean(), numpy.median(), numpy.std(), numpy.min(), and numpy.max() functions, respectively. Print and save these descriptive statistics to a text file using the print() and open() functions.

Save the average COF and time lists as new Excel files using the pandas.DataFrame() and pandas.to_excel() functions.

Read the data from the two Excel files using the pandas.read_excel() function and concatenate them into one DataFrame using the pandas.concat() function. Save the new DataFrame to an Excel file using the pandas.to_excel() function.

Load the data from the Excel file using the pandas.read_excel() function and assign it to a variable using the pandas.DataFrame() function. The DataFrame contains two columns: Average Time B (time in sec.) and Average Coefficient Of Friction B (COF).

Split the data into input (X) and output (y) variables using the pandas.DataFrame.iloc[] function. The input variable is the average time and the output variable is the average COF.

Split the data into training, validation, and testing sets using the sklearn.model_selection.train_test_split() function with a test size of 0.2 and a validation size of 0.25. Set a random state of 42 for reproducibility.

Define a parameter grid with the hyperparameters and their ranges for the SVR model using a dictionary. The hyperparameters are kernel, C, gamma, and epsilon.

Create a SVR object using the sklearn.svm.SVR() function with default parameters.

Create a grid search object with 5-fold cross-validation using the sklearn.model_selection.GridSearchCV() function with scoring=‘neg_mean_squared_error’ and n_jobs=-1.

Fit the grid search on the training data using the grid_search.fit() function.

Print the best parameters and score using the grid_search.best_params_ and grid_search.best_score_ attributes and the print() function.

Get the best estimator using the grid_search.best_estimator_ attribute and assign it to a variable.

Predict the COF for the test set using the best_estimator.predict() function and assign it to a variable.

Predict the COF for the validation set using the best_estimator.predict() function and assign it to a variable.

Calculate the R2 score for both sets using the sklearn.metrics.r2_score() function and assign it to a variable.

Calculate and store the performance metrics for both sets in a file using the numpy.sqrt(), sklearn.metrics.mean_squared_error(), sklearn.metrics.mean_absolute_error() functions and the open() function.

Create a figure object using the matplotlib.pyplot.figure() function.

Plot the actual vs predicted COF as a function of time using the matplotlib.pyplot.scatter() function.

Add labels, legend, title, gridlines, limits, and shadow effects to the plot using the matplotlib.pyplot.xlabel(), matplotlib.pyplot.ylabel(), matplotlib.pyplot.legend(), matplotlib.pyplot.title(), matplotlib.pyplot.grid(), matplotlib.pyplot.xlim(), matplotlib.pyplot.ylim(), and patheffects.SimpleLineShadow() functions.

Show and save the plot as an image file with dpi=500 in ‘png’ using the matplotlib.pyplot.show() and figure.savefig() functions.

Close plot material B using the matplotlib.pyplot.close() function.

Funding

This research was funded by Bulgarian National Science Fund, Project № КП-06-Н57/20 “Fabrication of new type of self-lubricating antifriction metal matrix composite materials with improved mechanical and tribological properties”.
