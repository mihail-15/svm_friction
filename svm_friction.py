# import the necessary packages
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import patheffects


# Code lines for Material E

# Read each .xlsx file containing the data for each dataset of the material 

file_E1 = pd.read_excel("E_1.xlsx") # read file for  E1
file_E2 = pd.read_excel("E_2.xlsx") # read file for  E2
file_E3 = pd.read_excel("E_3.xlsx") # read file for  E3

# Assign the dataset to a variable 

time_E1 = file_E1["TIME"] # time in seconds for  E1
friction_E1 = file_E1["FF"] # friction force in N for  E1
time_E2 = file_E2["TIME"] # time in seconds for  E2
friction_E2 = file_E2["FF"] # friction force in N for  E2
time_E3 = file_E3["TIME"] # time in seconds for  E3
friction_E3 = file_E3["FF"] # friction force in N for  E3

# Calculation of the COF for each dataset using the formula: COF = friction force / normal  force (50 N)


cof_E1 = friction_E1 / 50 # COF for  E1 
cof_E2 = friction_E2 / 50 # COF for  E2 
cof_E3 = friction_E3 / 50 # COF for  E3 


# Create an empty list to store the average COF 

avg_cof_E_list= [] 


# Create an empty list to store the average time 

avg_time_E_list= []  

# Average COF 
for i in range(0, len(time_E1)):
    avg_cof_E_list.append((cof_E1[i] + cof_E2[i] + cof_E3[i]) / 3)


# Average time
    avg_time_E_list.append((time_E1[i] + time_E2[i] + time_E3[i]) / 3)

# Save the list as a new excel file using pd.DataFrame() and pd.to_excel()

avg_cof_E_df= pd.DataFrame(avg_cof_E_list, columns=["Average Coefficient Of Friction E"]) 
avg_cof_E_df.to_excel("Average_COF_E.xlsx", index=False) 

avg_time_E_df= pd.DataFrame(avg_time_E_list, columns=["Average Time E"]) 
avg_time_E_df.to_excel("Average_time_E.xlsx", index=False) 


# read the data from the two excel files
avg_cof_E_df = pd.read_excel("Average_COF_E.xlsx") 
avg_time_E_df = pd.read_excel("Average_time_E.xlsx")

# concatenate the data from the two files into one DataFrame
merged_df = pd.concat([avg_time_E_df, avg_cof_E_df], axis=1)

# save the new DataFrame to an Excel file
merged_df.to_excel("COF_E.xlsx", index=False)


# load the data from the 'xlsx' file (for example "pred_COF_E.xlsx")
data = pd.read_excel(r"COF_E.xlsx")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# define a parameter grid with the hyperparameters and their ranges
param_grid = {
   'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
   'C': [0.1, 1, 10, 100], 
   'gamma': ['scale', 'auto'], 
   'epsilon': [0.01, 0.1, 0.5]
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

# calculate and store the performance metrics for both sets in a file (for example "E_performance_metrics.txt" or "SE_performance_metrics.txt")
with open('E_performance_metrics.txt','w') as f:
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

# x axis limit to 60
plt.xlim(0 ,450)

# y axis limit to 40
plt.ylim(0 ,0.6)

# gridlines to the plot
plt.grid(True)


# Add a title
plt.title("Material E", fontsize='18', fontweight='bold')

plt.show()

fig1 = plt.figure() 
plt.plot(X,y)
plt.xlim(0 ,450)
plt.ylim(0 ,0.6)
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
fig1.savefig('pred_COF_E.png', dpi=500)


# Close plot E
plt.close(fig1)

# create a DataFrame from the variables
df = pd.DataFrame({"Actual test": y_test, "Predicted test": y_pred, "Actual val": y_val, "Predicted val": y_val_pred})
# save the DataFrame to an Excel file
df.to_excel("test_val_data_E.xlsx", index=False)




# Code lines for Material SE


# Read each .xlsx file containing the data for each dataset of the material 

file_SE1 = pd.read_excel("SE_1.xlsx") # read file for  SE1
file_SE2 = pd.read_excel("SE_2.xlsx") # read file for  SE2
file_SE3 = pd.read_excel("SE_3.xlsx") # read file for  SE3

# Assign the dataset to a variable 

time_SE1 = file_SE1["TIME"] # time in seconds for  SE1
friction_SE1 = file_SE1["FF"] # friction force in N for  SE1
time_SE2 = file_SE2["TIME"] # time in seconds for  SE2
friction_SE2 = file_SE2["FF"] # friction force in N for  SE2
time_SE3 = file_SE3["TIME"] # time in seconds for  SE3
friction_SE3 = file_SE3["FF"] # friction force in N for  SE3

# Calculation of the COF for each dataset using the formula: COF = friction force / normal  force (50 N)


cof_SE1 = friction_SE1 / 50 # COF for  SE1 
cof_SE2 = friction_SE2 / 50 # COF for  SE2 
cof_SE3 = friction_SE3 / 50 # COF for  SE3 


# Create an empty list to store the average COF 

avg_cof_SE_list= [] 


# Create an empty list to store the average time 

avg_time_SE_list= []  

# Average COF 
for i in range(0, len(time_SE1)):
    avg_cof_SE_list.append((cof_SE1[i] + cof_SE2[i] + cof_SE3[i]) / 3)


# Average time
    avg_time_SE_list.append((time_SE1[i] + time_SE2[i] + time_SE3[i]) / 3)

# Save the list as a new excel file using pd.DataFrame() and pd.to_excel()

avg_cof_SE_df= pd.DataFrame(avg_cof_SE_list, columns=["Average Coefficient Of Friction SE"]) 
avg_cof_SE_df.to_excel("Average_COF_SE.xlsx", index=False) 

avg_time_SE_df= pd.DataFrame(avg_time_SE_list, columns=["Average Time SE"]) 
avg_time_SE_df.to_excel("Average_time_SE.xlsx", index=False) 


# read the data from the two excel files
avg_cof_SE_df = pd.read_excel("Average_COF_SE.xlsx") 
avg_time_SE_df = pd.read_excel("Average_time_SE.xlsx")

# concatenate the data from the two files into one DataFrame
merged_df = pd.concat([avg_time_SE_df, avg_cof_SE_df], axis=1)

# save the new DataFrame to an Excel file
merged_df.to_excel("COF_SE.xlsx", index=False)


# load the data from the 'xlsx' file (for example "pred_COF_E.xlsx")
data = pd.read_excel(r"COF_SE.xlsx")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# define a parameter grid with the hyperparameters and their ranges

param_grid = {
   'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
   'C': [0.1, 1, 10, 100], 
   'gamma': ['scale', 'auto'], 
   'epsilon': [0.01, 0.1, 0.5]
}


# create a SVR
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

# calculate and store the performance metrics for both sets in a file (for example "E_performance_metrics.txt" or "SE_performance_metrics.txt")
with open('SE_performance_metrics.txt','w') as f:
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

# Create figure 2
fig2 = plt.figure()

# Shadow effect objects with different transparency and smaller linewidth
pe2 = [patheffects.SimpleLineShadow(offset=(0.5,-0.5), alpha=0.4), patheffects.Normal()]

# Plot of the actual vs predicted COF as a function of time
plt.scatter(X_test[:, 0], y_test,color='cyan',label='Actual test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.scatter(X_test[:, 0], y_pred,color='orange',label='Predicted test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.scatter(X_val[:, 0], y_val,color='green',label='Actual val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.scatter(X_val[:, 0], y_val_pred,color='magenta',label='Predicted val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.xlabel('Time, s', fontsize='15', fontweight='bold')
plt.ylabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.legend(shadow=True, prop={'size':'12'}, loc='lower right')

# x axis limit to 60
plt.xlim(0 ,450)

# y axis limit to 40
plt.ylim(0 ,0.6)

# gridlines to the plot
plt.grid(True)

# Add a title
plt.title("Material SE", fontsize='18', fontweight='bold')

plt.show()

fig2 = plt.figure() 
plt.plot(X,y)
plt.xlim(0 ,450)
plt.ylim(0 ,0.6)
plt.grid(True)

# Plot of the actual vs predicted COF as a function of time
plt.scatter(X_test[:, 0], y_test,color='cyan',label='Actual test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.scatter(X_test[:, 0], y_pred,color='orange',label='Predicted test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.scatter(X_val[:, 0], y_val,color='green',label='Actual val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.scatter(X_val[:, 0], y_val_pred,color='magenta',label='Predicted val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.xlabel('Time, s', fontsize='15', fontweight='bold')
plt.ylabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.legend(loc='lower right')


# Save the plot with dpi=500 in 'png'
fig2.savefig('pred_COF_SE.png', dpi=500)




# create a DataFrame from the variables
df = pd.DataFrame({"Actual test": y_test, "Predicted test": y_pred, "Actual val": y_val, "Predicted val": y_val_pred})
# save the DataFrame to an Excel file
df.to_excel("test_val_data_SE.xlsx", index=False)
