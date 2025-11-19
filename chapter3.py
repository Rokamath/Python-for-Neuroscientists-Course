import numpy as np

# Print sqrt function
# print(np.sqrt(36))

'''Arrays'''

'''Printing array type'''
a = np.array([1, 3, 5])
#print(type(a))

'''Print out an array'''
b = np.array([1, 2, "hello"])
#print(b)

'''Print the shape of the array in Rows and Columns'''
c = np.array(
    [[1,2],
     [3,4]]
    )
#print(c.shape)

'''Vector operations'''

#lists
temperature = [18.0, 21.5, 21.0, 21.0, 18.8, 17.6, 20.9, 20.0]
temperature = list(map(lambda x: x * 9/5 + 32, temperature))
#print(temperature)

# numpy array
temperature_array = np.array([18.0, 21.5, 21.0, 21.0, 18.8, 17.6, 20.9, 20.0])
temperature_array = temperature_array * 9/5 + 32
#print(temperature_array)

'''Linear algebra (1) '''

A = np.array(
    [[6, 1, 1],
     [4, -2, 5],
     [2, 8, 7]]
    )

# Rank of a matrix 
'''Max number of linearly independent row(or column) vectors of the matrix'''
#print("Rank of A:", np.linalg.matrix_rank(A))
 
'''Defined as the sum of its Diagonal elements'''
# Trace of matrix A
#print("Trace of A:", np.trace(A))
 
# Determinant of a matrix
#print("Determinant of A:", np.linalg.det(A))
 
# Inverse of matrix A
#print("Inverse of A:\n", np.linalg.inv(A))

# Size of matrix A
#print("Size of the array: ", np.size(A))

'''Linear algebra (2)'''

'''{1 ** x1 + 2 ** x2 = 10}'''
'''{3 ** x1 + 4 ** x2 = 20}'''
C = np.array(
    [[1,2],
     [3,4]])

D = np.array([10, 20])

x = np.linalg.solve(C,D)
#print(x) # Prints x1 = 0 and x2 = 5

'''Exercise. 
You have a sample from EEG data from Cavanagh et al. (2019) study. 
This sample consists of 

around 10 seconds of data measurements 
sampled at 500 Hz frequency 
measured at 66 channels.'''

'''
Your task is:

Find the index of the C1 electrode in eeg['ch_names'] array.
Use the location of the C1 electrode and indexes (1000:2001) for the time axis to slice the signal for the channel C1 from 2s to 4s of recording. Note that the resulting array will have the shape (1, 1001).
Find the maximum, minimum, and mean values of the resulting array c1_data.
Find the indexes that correspond to these values in the c1_data array using np.where() function.
Print out the results using f-string. Convert values to micro Volts and round up to the 2nd digit.
'''



'''eeg['ch_names']: numpy array of shape (66,). # Represents the channel names. 
eeg['data']: numpy array of shape (66, 50001). # Represents the actual data in a (channel, time) format.
eeg['srate']: int. # Represents the sampling rate [Hz].
eeg['times']: numpy array of shape (50001,). # Represents the time vector [s].
'''

import pickle
import matplotlib.pyplot as plt

# load the pickled file
with open("exercises/data/eeg_sample.pickle", mode="rb") as f:
    eeg = pickle.load(file=f)

# filter the elements in the array to find the location (index) of channel C1
c1_loc = eeg['ch_names'] == 'C1'
# slice the array with the data using the index of C1 location and
# 1000:2001 bounds for time axis (bounds correspond to 2 and 4s or recordings)
c1_data = eeg['data'][c1_loc, 1000:2001]

# find the max, min and mean value of the c1_data array
max_v = c1_data.max()
min_v = c1_data.min()
mean_v = c1_data.mean()

# find the indexes (time point) of max and min values
t_max_v = np.where(c1_data[0, :] == max_v)
t_min_v = np.where(c1_data[0, :] == min_v)

# print out the results using f-string.
# convert volts to microvolts and round up the outcome values to the 2 digit
#print(f"Max value: {max_v * 1e06:.2f}, microV \
#    \n Min value: {min_v * 1e06:.2f}, microV \
#    \n Average value: {mean_v * 1e06:.2f}, microV")

# plot the results
plt.figure(figsize=(10,5), facecolor="white")
plt.plot(c1_data[0, :], c='black', label='Signal')
plt.plot(t_max_v, max_v, 'o', c='red', label='Maximum Value')
plt.plot(t_min_v, min_v, 'o', c='blue', label='Minimum Value')
plt.axhline(y=mean_v, linestyle='dashed', lw=1, label='Average Value')
plt.title('Sample of the Signal', fontweight='bold', fontsize=18)
plt.xlabel('Time point')
plt.ylabel('Voltage, [V]')
plt.legend()
#plt.show()

''' You could notice that solution using c1_data[0, :] for finding the time points of max and min values. Why did we include only 0 value for the 1-st dimension in the slicing? Let’s take a look at two cases: '''

t_1 = np.where(c1_data == max_v)
t_2 = np.where(c1_data[0, :] == max_v)
#print(t_1) # (array([0]), array([524]))
#print(t_2) # (array([524]),)

'''In the first case for t_1, function np.where() found the maximum value in the whole array and returned a tuple, which tells us that the maximum element is located on the 0th index for the first dimension and 524-th index for the second dimension. 

But since we know that our data doesn’t have any other indexes in the first dimension, we can search for the index using 0 index as a slicing value to avoid redundant results as we did for t_2 example. 

In this case, we also reshape the c1_data array from a 2-D array with the shape (1, 1001) to a 1-D array with the shape (1001,). '''

'''Linear Regression'''

'''Transpose matrix can be found using method ''' # .transpose() or .T;
''' Inverse matrix can be found using ''' # np.linalg.inv(<array>);
'''Matrix multiplication can be done using ''' # np.matmul(<array1>, <array2>) or <array1>@<array2>.

X = np.load("exercises/data/X.npy")
y = np.load("exercises/data/y.npy")

b = np.linalg.inv(X.T @ X) @ X.T @ y

#print(f"Linear Regression Model: Accuracy = {b[0]:.2f} + {b[1]:.2f}*TAI")

#print(X)
#print(y)

'''Build a Linear Regression model using the 
Ordinary Least Squares method'''

'''
OLS Least Square Method
Accuracy = b0 + b1 * TAI
    b0 = intercept
    b1 = slope parameter
    b coefficients = (X**T * X)**-1 * X**T * y
'''
#print(X.shape) # 20,2
#print(y.shape) # 20,

'''Exercise 2. 
Now that you have found b coefficients, 
what is the value of root mean squared error (RMSE)? '''

# ypred - ytrue = residual (error)

''' To find RMSE, you have to :
Find the predicted value of accuracy (y) for each TAI score (X). Keep in mind, that X array has two columns, the first column consists of ones and the second column is the actual TAI scores.

Take the difference between the predicted and actual value of y (residuals) and square it.

Take the average of all squared residual values and get a square root of that value.'''

X = np.load("exercises/data/X.npy")
y = np.load("exercises/data/y.npy")
b = np.load("exercises/data/b.npy")

y_pred = b[0] + b[1]*X[:,1]        # the predicted values /// # Takes data from second column, X[r,c] > X[,1]
residuals_sq = (y_pred - y)**2           # squared residuals
RMSE = np.sqrt(np.mean(residuals_sq))                    # square root of average residuals squared

#print(f"RMSE = {RMSE: .2f}")

'''Working with table data with Pandas'''

import pandas as pd

temperature = [18.0, 21.5, 21.0, 21.0, 18.8, 17.6, 20.9, 20.0]
temperature_series = pd.Series(temperature)
#print(temperature_series)

''' Pandas slicing an index 
[<start index>:<end index>:<step>] ''' # includes 0

#print(temperature_series[2:4]) # slicing

temperature_series = temperature_series * 9/5 + 32 # numerical operations to whole series
#print(temperature_series)

#print(temperature_series.mean()) # useful methods

''' Pandas Dataframe'''
dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv")
#print(dementia_df) prints full dataframe
#print(dementia_df.info()) # prints info first 5 and last 5 columns

#print(dementia_df['Age']) # prints only age column Series
#print(dementia_df[['Age', 'Educ', 'SES']]) # prints DataFrame

'''Rows'''
#print(dementia_df.iloc[0]) # Only print first Row
#print(dementia_df.iloc[-1]) # Prints last row

'''Columns'''
#print(dementia_df.iloc[:,0]) # first Column of data frame (id)
#print(dementia_df.iloc[:,1]) # second Column of data frame (M/F)
#print(dementia_df.iloc[:,-1]) # last Column of data frame (Delay)

'''
DataFrames have a bit specific way of indexing. Remember that you can index by the columns and rows at the same time. There are two options:

Index the the index (or integer-location). In this case we use <DataFrame>.iloc[<row position(s)>, <column position(s)>]. You can specify one position as an integer, or multiple positions as a list or use slicing <position1>:<position2>. All the columns/rows will be selected (excluding the position2) in the slice.

Index by the name (or location). In this case, we use <DataFrame>.loc[<row name(s)>, <column name(s)>]. You can specify one name as a string, multiple names as a list, or use slicing <name1>:<name2>. All the columns/rows will be selected (including the name2) in the slice. 

Also, you can specify the Boolean list index where all the rows/columns that correspond to the True value will be included.
'''

'''Filtering data in Pandas Dataframe'''

condition = (dementia_df['M/F'] == 'F') & (dementia_df['Age'] > 60) # Prints Females which are above 60 years old
#print(dementia_df[condition])

'''Filtering within one column'''

#print(dementia_df['ID'][dementia_df['nWBV'].between(0.7, 0.8)]) # Prints normalized whole-brain volume (nWBV) should be in a range between 0.7 and 0.8.

'''Data Aggregation'''
''' In this example, we want to see 
        the minimum estimated total intracranial volume (eTIV) 
        and average normalized whole-brain volume (nWBV) 
            for 
                each gender 
                and clinical dementia rating (CDR)'''

'''Clinical Dementia Rating: 0 = no dementia, 0.5 = very mild AD, 1 = mild AD, 2 = moderate AD'''

''' Group by a list of column names. 
If you wanted to split just by one column, you could 
    specify just a string with a column name, for example, by="CDR".

We apply aggregation method and specify a dictionary in a following way: 
    
    ''' ### {"<column name>": "<aggregation function>"}. 

'''If you wanted to apply multiple functions on the same column: 
    
    you could specify a list, for example, ''' # {"eTIV": ["min", "max"]}.

#print(dementia_df.groupby(by=["CDR", "M/F"]).agg({"eTIV": "min", "nWBV": "mean"})) # Prints frame grouped by CDR and Gender columns only and aggregates Min eTIV and Mean nWBV

'''Exercises'''

'''  11  Delay   20 non-null     float64 '''
 # Entries - Non-Null = Missing Values

'''Exercise 1. 
 What is the average socioeconomic status (SES) for subjects without dementia? 
 Do this in two methods (that will lead to the same outcome): through selection of the column and filtering condition separately and by using .loc operator.

Clinical Dementia Rating column (CDR) is represented in a following way: 0 = no dementia, 0.5 = very mild AD, 1 = mild AD, 2 = moderate AD.

Socioeconomic status is assessed by the Hollingshead Index of Social Position and classified into categories from 1 (highest status) to 5 (lowest status).'''

dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv")

avg_ses1 = dementia_df[dementia_df['CDR'] == 0.0]['SES'].mean() # Filters where CDR is 0 and finds the mean of SES ''
avg_ses2 = dementia_df.loc[dementia_df['CDR'] == 0.0, "SES"].mean() # Locates using intergers where CDR is 0 and in same line the mean of SES ""

#print(avg_ses1 == avg_ses2)  # check that values are the same
#print(round(avg_ses1, 1))

'''Exercise 2'''

'''For each outcome of clinical dementia rating (CDR) 
    get a count of observations 
    the median of the normalized whole-brain volume (nWBV) 
    Rename the columns in aggregated DataFrame, so they are more representative.

Hint! <DataFrame>.rename() method takes an argument columns to change the column names. You should pass a dictionary in a dictionary {"<old name>": "<new name>"}. inplace=True will overwrite the DataFrame (save the new column names).'''

dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv")

agg_df = dementia_df.groupby(by=["CDR"]).agg({"ID": "count", "nWBV": "median"}) ## Grouped by CDR, aggregated by counting # of OBV and found median nWBV
#print("Original:")
#print(agg_df)

agg_df.rename(columns={"ID": "Count", "nWBV": "nWBV_median"}, inplace=True) ## Renamed columns from ID > Count and nWBV > nWBV_median
#print("Renamed:")
#print(agg_df)

'''Types of Joins in Pandas
Type	Description	Scheme

Inner Join	Returns records that have matching values in both tables	

Left Join	Returns all records from the left table,
and the matched records from the right table	

Right Join	Returns all records from the right table,
and the matched records from the left table	

Full Join	Returns all records when there is a match
in either left or right table	'''

'''Note that it's important that you have a shared column to join the data.'''

#table1 = pd.DataFrame(
#    {'Id': [1, 2, 3, 4],
#    'Name': ['Bob', 'Jack', 'Jill', 'Ben']})

#table2 = pd.DataFrame(
#    {'Id': [1,1,3,5,7,7],
#    'Occupation': ['IT', 'Finance', 'IT', 'Healthcare', 'Agriculture', 'Finance']})

#pd.merge(
#    left=table1, right=table2, ## SPecify which Table is LEFT and RIGHT
#    on='Id', # or left_on='Id', right_on='Id', 
#    how='left') # How to Join LEFT or RIGHT or INNER or FULL

'''Exercise 1. 
You have two DataFrames loaded in. 
    One has IDs of patients and the breast cancer status (malignant or benign). 
    The other one has IDs of patients and some features for the cell nucleus.

radius_mean: mean of distances from the center to points on the perimeter;
texture_mean: standard deviation of gray-scale values;
perimeter: mean size of the core tumor.

Is the average radius_mean value is greater for malignant type?'''

#table_1 = pd.read_json("exercises/data/table1.json")
#table_2 = pd.read_json("exercises/data/table2.json")

#joined_table = pd.merge(left=table_1, right=table_2, how='right', ## merge right
#                        left_on='id', right_on='ID') # columns to link

#radius_mean_benign = joined_table[joined_table['diagnosis'] == 'benign']['radius_mean'].mean()
#radius_mean_malignant = joined_table[joined_table['diagnosis'] == 'malignant']['radius_mean'].mean()

#print(radius_mean_malignant > radius_mean_benign)

'''Exercise 2. This time get all the features for 
the cell nucleus and 
label them with the type of cancer. 

When the cancer type is not specified mark it as “unknown”. 
Do not change missing values in other columns.

The joined DataFrame will have two columns "id" and "ID". Keep only the first column.

Hint! To replace the missing value you can use .fillna() method. It can be applied on Series or DataFrame.'''

table_1 = pd.read_json("exercises/data/table1.json")
table_2 = pd.read_json("exercises/data/table2.json")

joined_table = pd.merge(left=table_1, right=table_2, how='right', ## merge right
                        left_on='id', right_on='ID') # join two tables together

joined_table.drop(labels="ID", axis=1, inplace=True) # drop the redundant column and choose index of column as 1
joined_table["diagnosis"] = joined_table["diagnosis"].fillna("unknown")#  replace the missing values in a column with Unknown

#print(joined_table)