# import the necessary packages
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import patheffects


# Code lines for AMMCs (B)

# Read each .xlsx file containing the data for each dataset of the AMMCs 

file_B1 = pd.read_excel("B_1.xlsx") # read file for  B1
file_B2 = pd.read_excel("B_2.xlsx") # read file for  B2
file_B3 = pd.read_excel("B_3.xlsx") # read file for  B3

# Assign the dataset to a variable 

time_B1 = file_B1["TIME"] # time in sec. for B1
friction_B1 = file_B1["FF"] # friction force in N  for B1
time_B2 = file_B2["TIME"] # time in sec. for B2
friction_B2 = file_B2["FF"] # friction force in N  for B2
time_B3 = file_B3["TIME"] # time in sec. for B3
friction_B3 = file_B3["FF"] # friction force in N  for B3

# Calculation of the COF for each dataset using the formula: COF = friction force / normal  force (40 N)


cof_B1 = friction_B1 / 100 # COF for  B1 
cof_B2 = friction_B2 / 100 # COF for  B2 
cof_B3 = friction_B3 / 100 # COF for  B3 



# Create an empty list to store the average COF 

avg_cof_B_list= [] 


# Create an empty list to store the average time 

avg_time_B_list= []  

# Average COF 
for i in range(0, len(time_B1)):
    avg_cof_B_list.append((cof_B1[i] + cof_B2[i] + cof_B3[i]) / 3)


# Average time
    avg_time_B_list.append((time_B1[i] + time_B2[i] + time_B3[i]) / 3)


# For material B
mean_cof_B = np.mean(avg_cof_B_list) # calculate the mean
median_cof_B = np.median(avg_cof_B_list) # calculate the median
std_cof_B = np.std(avg_cof_B_list) # calculate the standard deviation
min_cof_B = np.min(avg_cof_B_list) # calculate the minimum
max_cof_B = np.max(avg_cof_B_list) # calculate the maximum

# Print the descriptive statistics for AMMCs B
print("Descriptive statistics for material B:")
print("Mean: {:.4f}".format(mean_cof_B))
print("Median: {:.4f}".format(median_cof_B))
print("Standard deviation: {:.4f}".format(std_cof_B))
print("Minimum: {:.4f}".format(min_cof_B))
print("Maximum: {:.4f}".format(max_cof_B))

# Save the descriptive statistics to a file
with open('B_descriptive_statistics.txt','w') as f:
 f.write('Descriptive statistics for material B:\n')
 f.write('Mean: {:.4f}\n'.format(mean_cof_B))
 f.write('Median: {:.4f}\n'.format(median_cof_B))
 f.write('Standard deviation: {:.4f}\n'.format(std_cof_B))
 f.write('Minimum: {:.4f}\n'.format(min_cof_B))
 f.write('Maximum: {:.4f}\n'.format(max_cof_B))
 f.close()


# Save the list as a new excel file using pd.DataFrame() and pd.to_excel()

avg_cof_B_df= pd.DataFrame(avg_cof_B_list, columns=["Average Coefficient Of Friction B"]) 
avg_cof_B_df.to_excel("Average_COF_B.xlsx", index=False) 

avg_time_B_df= pd.DataFrame(avg_time_B_list, columns=["Average Time B"]) 
avg_time_B_df.to_excel("Average_time_B.xlsx", index=False) 


# read the data from the two excel files
avg_cof_B_df = pd.read_excel("Average_COF_B.xlsx") 
avg_time_B_df = pd.read_excel("Average_time_B.xlsx")

# concatenate the data from the two files into one DataFrame
merged_df = pd.concat([avg_time_B_df, avg_cof_B_df], axis=1)

# save the new DataFrame to an Excel file
merged_df.to_excel("COF_B.xlsx", index=False)


# load the data from the 'xlsx' file (in our case "pred_COF_B.xlsx")
data = pd.read_excel(r"COF_B.xlsx")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# define a parameter grid with the hyperparameters and their ranges
param_grid = {
   'kernel': ['linear', 'rbf'],   # ['linear', 'poly', 'rbf', 'sigmoid'],  
                                  # 'degree': [2, 3],   # 'degree': [2, 3, 4, 5], only with poly
   'C': [0.1, 1, 10, 100],         # [0.1, 1, 10],         # [0.5, 1, 10, 100], 
   'gamma': [1e-3, 1e-4],                  # [1e-3, 1e-4],          # [0.001, 0.01]  # ['scale', 'auto'],
   'epsilon': [0.01, 0.1, 0.5]     # [0.01, 0.1, 0.5]  # [0.05, 0.5]
}

# create a Support Vector Machine
svr = SVR()

# create a grid search object with 5-fold cross-validation
gs = GridSearchCV(svr, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# fit the grid search on the training data

gs.fit(X_train, y_train)


# print the best parameters and score
print("Best parameters: ", gs.best_params_)
print("Best score: ", gs.best_score_)

# get the best estimator
dt_best = gs.best_estimator_

# predict the coefficient of friction for the test set

y_pred = dt_best.predict(X_test)

# predict the coefficient of friction for the validation set
y_val_pred = dt_best.predict(X_val)

# calculate the R2 score for both sets
r2_test = r2_score(y_test,y_pred)
r2_val = r2_score(y_val,y_val_pred)

# calculate and store the performance metrics for both sets in a file (for AMMCs "B_performance_metrics.txt")
with open('B_performance_metrics.txt','w') as f:
 f.write('Test set performance metrics:\n')
 f.write('R2 score: {:.4f}\n'.format(r2_test))
 f.write('RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
 f.write('MSE: {:.4f}\n'.format(mean_squared_error(y_test,y_pred)))
 f.write('MAE: {:.4f}\n'.format(mean_absolute_error(y_test,y_pred)))
 f.write('Validation set performance metrics:\n')
 f.write('R2 score: {:.4f}\n'.format(r2_val))
 f.write('RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_val,y_val_pred))))
 f.write('MSE: {:.4f}\n'.format(mean_squared_error(y_val,y_val_pred)))
 f.write('MAE: {:.4f}\n'.format(mean_absolute_error(y_val,y_val_pred)))
 f.close()


# Create figure 1
fig1 = plt.figure()



# Shadow effect objects with different transparency and smaller linewidth
pe1 = [patheffects.SimpleLineShadow(offset=(0.5,-0.5), alpha=0.4), patheffects.Normal()]

# Plot of the actual vs predicted COF as a function of time
plt.scatter(X_test[:, 0], y_test,color='cyan',label='Actual test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_test[:, 0], y_pred,color='orange',label='Predicted test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_val[:, 0], y_val,color='green',label='Actual val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_val[:, 0], y_val_pred,color='magenta',label='Predicted val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.xlabel('Time, s', fontsize='15', fontweight='bold')
plt.ylabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.legend(shadow=True, prop={'size':'12'}, loc='lower right')

# x axis limit 
plt.xlim(0 ,450)

# y axis limit 
plt.ylim(0 ,0.7)

# gridlines to the plot
plt.grid(True)


# Add a title
plt.title("AMMCs", fontsize='18', fontweight='bold')

plt.show()

fig1 = plt.figure() 
plt.plot(X,y)
plt.xlim(0 ,450)
plt.ylim(0 ,0.7)
plt.grid(True)

# Plot of the actual vs predicted COF as a function of time
plt.scatter(X_test[:, 0], y_test,color='cyan',label='Actual test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_test[:, 0], y_pred,color='orange',label='Predicted test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_val[:, 0], y_val,color='green',label='Actual val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_val[:, 0], y_val_pred,color='magenta',label='Predicted val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.xlabel('Time, s', fontsize='15', fontweight='bold')
plt.ylabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.legend(loc='lower right')


# Save the plot with dpi=500 in 'png'
fig1.savefig('pred_COF_B.png', dpi=500)


# Close plot material B
plt.close(fig1)

# create a DataFrame from the variables
df = pd.DataFrame({"Actual test": y_test, "Predicted test": y_pred, "Actual val": y_val, "Predicted val": y_val_pred})
# save the DataFrame to an Excel file
df.to_excel("test_val_data_B.xlsx", index=False)






# For material B
fig3 = plt.figure() # assign a variable to the figure object
plt.hist(avg_cof_B_list, bins=20, color='blue', edgecolor='black') # plot the histogram
plt.xlabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.ylabel('Frequency, -', fontsize='15', fontweight='bold')
plt.title('Histogram of COF for AMMCs', fontsize='18', fontweight='bold')
plt.show()

fig3.savefig('hist_COF_B.png', dpi=500) # save the figure to a file

fig4 = plt.figure() # assign a variable to the figure object
plt.boxplot(avg_cof_B_list, vert=False, showmeans=True) # plot the boxplot
plt.xlabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.title('Boxplot of COF for AMMCs', fontsize='18', fontweight='bold')
plt.show()

fig4.savefig('box_COF_B.png', dpi=500) # save the figure to a file

