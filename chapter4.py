'''Data Visualization'''

'''Data visualization is the graphic representation of data. It involves producing images that communicate relationships among the represented data to viewers of the images. This communication is achieved through the use of a systematic mapping between graphic marks and data values in the creation of the visualization. This mapping establishes how data values will be represented visually, determining how and to what extent a property of a graphic mark, such as size or color, will change to reflect change in the value of a datum.
'''

'''How to create a meaningful plot'''

'''Rule of thumb:
Think about what answer should you visualization answer.
Choose the right chart for your data.
Make it clear and human-readable.
Don’t put too much information on one chart.
Describe it with titles, labels, and annotations.
Don’t go crazy with colors.'''

import matplotlib.pyplot as plt
import numpy as np

#x = np.linspace(start=0, stop=2*np.pi, num=100)
#y_sin = np.sin(x)

#z = np.linspace(start=0, stop=2*np.pi, num=100)
#z_cos = np.cos(x)

#plt.figure() # start
#plt.plot(x, y_sin)
#plt.plot(z,z_cos)
#plt.title("cos and sin")
#plt.xlabel("Degrees")
#plt.ylabel("Amplitude")
#plt.show()   # end

#y_cos = np.cos(x)

#plt.figure() # start
#plt.plot(x, y_sin, 'o--', color='r', label='Sine') # 'o--' - scatter plot connected with a dashed line; 
                                                   # with one letter ("g", meaning green),("#ffffff") or RGB color as a list ((0.1, 0.2, 0.5) 
                                                   # label is responsible for assigning a name to an object on a legend.
#plt.plot(x, y_cos, color='black', label='Cosine')
#plt.axhline(y=0, linewidth=1, color='#42f5b0', linestyle='dashed', label='Zero') # axHline/ axVline creates a horizontal/vertical through axis
#plt.title("My First Plot", fontsize=18)
#plt.xlabel("This is x axis")
#plt.ylabel("This is y axis")
#plt.legend()
#plt.show()   # end

'''Area chart'''
#plt.figure()
#plt.plot(x, y_sin, x, y_cos)
#plt.fill_between(x, y_sin, alpha=0.5) # Use plt.fill_between() function - alpha is for opacity (0: object is transparent, 1: full color).
#plt.fill_between(x, y_cos, alpha=0.5)
#plt.xlabel('x axis')
#plt.ylabel('y axis')
#plt.title("Sine/Cosine Area Chart", fontsize=18)
#plt.show()

import pandas as pd

'''Bar Plot'''
#df = pd.DataFrame(
#    {"month": ["Jan", "Feb", "Mar", "Apr", "May"],
#     "value": [100, 130, 200, 120, 140]})
#df.sort_values(by="value", ascending=False, inplace=True) # set order of bars

#plt.figure()
#plt.bar(df["month"], df["value"])
#plt.title("Bar Chart", fontsize=18)
#plt.xlabel("Month")
#plt.ylabel("Value")
#plt.show()

'''Seaborn'''
import seaborn as sns 

'''Seaborn Scatterplot'''
#df = pd.DataFrame(
#    {"height": [167, 189, 170, 175, 190, 183],
#     "weight": [65, 78, 60, 68, 79, 72]})
#plt.figure()
#sns.scatterplot(data=df, x="height", y="weight") # Seaborn takes the axis labels from column names
#plt.title("Height vs Weight", fontsize=18)
#plt.show()

'''Subplots'''
#df = pd.DataFrame(
#    {"height": [167, 189, 170, 175, 190, 183],
#     "weight": [65, 78, 60, 68, 79, 72],
#     "gender": ["M", "F", "F", "F", "M", "M"]})

#plt.figure(figsize=(10,4))
#plt.subplot(1,2,1) # total number of rows for the final plot; total number of columns for the final plot; index of the plot that is currently modified. Ordering goes left to right, top to bottom. - 1R and 2C
#sns.scatterplot(data=df, x="height", y="weight", hue="gender") #hue allows adding a grouping variable;
#plt.title("Height vs Weight", fontsize=18)

#plt.subplot(1,2,2)  # total number of rows for the final plot; total number of columns for the final plot; index of the plot that is currently modified.
#sns.barplot(data=df, x="weight", y="weight")
#plt.title("Average Height by Gender", fontsize=18)
#plt.show()

'''Exercise 1. Breast cancer diagnostic data set'''

#Load the data with breast cancer observations (path to the file "exercises/data/breast_cancer.csv");
#Make a scatter plot of radius_mean on the x axis against texture_mean on the y axis along with the regression line. You can do this in one line of code using sns.regplot() function. Specify standard deviation as an error term (x_ci="sd");
#Add a vertical line with average radius and horizontal line with average texture. Set both of them to be "dashed", "black" colored, and of width 1.

# read in the data
#cancer_df = pd.read_csv("exercises/data/breast_cancer.csv")

#plt.figure(facecolor="white")
# regression plot
#sns.regplot(x="radius_mean", y="texture_mean", data=cancer_df, x_ci='sd')
# add the vertical line with the average radius_mean
#plt.axvline(x=cancer_df["radius_mean"].mean(), color="black", linewidth=1, linestyle="dashed")
# add the horizontal line with the average texture_mean
#plt.axhline(y=cancer_df["texture_mean"].mean(), color="black", linewidth=1, linestyle="dashed")
#plt.title("Radius vs Texture")
# show the figure
#plt.show()

'''Exercise 2. 
It is always a good idea to check for a relationship among variables in the data before applying Machine Learning algorithms. 
One of the possible ways to do this is to check the correlation. 
Looking at the raw numbers of correlation might be not that productive and that is when heatmap becomes handy.

Load the data with breast cancer observations (path to the file "exercises/data/breast_cancer.csv");
Select only columns the average values (with the "mean" in the name);
Create a correlation matrix and store it to corr_matrix variable;
Pass the corr_matrix into the sns.heatmap() function to create a heatmap.
'''

# read in the data
#cancer_df = pd.read_csv("exercises/data/breast_cancer.csv")

# select the columns with 'mean' in the name
#selected_columns = list(filter(lambda x: "mean" in x, cancer_df.columns)) #a lambda function which sorts columnds if that WORD is in the column name

# find the correlations
#corr_matrix = cancer_df[selected_columns].corr()

# make a plot
#plt.figure(figsize=(8,7), facecolor='white')
#sns.heatmap(data=corr_matrix, cmap="YlGnBu")
#plt.title("Correlation Among Variables")
#plt.show()

''' Exploratory Data Analysis
In statistics, exploratory data analysis is an approach to analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. Exploratory data analysis was promoted by John Tukey to encourage statisticians to explore the data, and possibly formulate hypotheses that could lead to new data collection and experiments.
'''

#Load the data with dementia patients’ observations (path to the file "exercises/data/oasis_cross-sectional.csv").
#Create 4 grouped bar charts at the same figure (using plt.subplot()). Variables to plot:

#y axis: age ("Age"), years of education ("Educ"), socioeconomic status ("SES"), Mini-Mental State Examination score ("MMSE");
#x axis: Clinical Dementia Rating ("CDR")
#group (hue argument) by gender ("M/F")
#CDR classification:

#0 = Normal
#0.5 = Very Mild Dementia
#1 = Mild Dementia
#2 = Moderate Dementia

#Additionally, create a summary_stats DataFrame with the count of observations ("ID" column) and aggregated values (mean and standard deviation of these four variables).

# read in the data
#dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv")

#plt.figure(figsize=(10,7), facecolor="white")

#(1,2,1) # total number of rows for the final plot; total number of columns for the final plot; index of the plot that is currently modified. Ordering goes left to right, top to bottom

# add first figure to the 2x2 plot with age ("Age")
#plt.subplot(2,2,1)
#sns.barplot(
#    x="CDR", data=dementia_df, y="Age",
#    hue= "M/F", ci="sd", color="lightblue")
#plt.title("Age (Mean ± SD)")

# add second figure to the 2x2 plot
# with years of education ("Educ")
#plt.subplot(2,2,2)
#sns.barplot(
#    x="Educ", data=dementia_df, y="CDR",
#    hue= "M/F", ci="sd", color="lightblue")
#plt.title("Years of education (Mean ± SD)")

# add third figure to the 2x2 plot
# with socioeconomic status ("SES")
#plt.subplot(2,2,3)
#sns.barplot(
#    x="CDR", data=dementia_df, y="SES",
#    hue= "M/F", ci="sd", color="lightblue")
#plt.title("Socioeconomic status (Mean ± SD)")

# add third fourth to the 2x2 plot
# with Mini-Mental State Examination score ("MMSE")
#plt.subplot(2,2,4)
#sns.barplot(
#    x="CDR", data=dementia_df, y="MMSE",
#    hue= "M/F", ci="sd", color="lightblue")
#plt.title("Mini-Mental State Examination score (Mean ± SD)")

#plt.tight_layout() # adjust the padding between and around subplots
#plt.show()

#print("Summary statistics:")

#Additionally, create a summary_stats DataFrame with the count of observations ("ID" column) and aggregated values (mean and standard deviation of these four variables).

#avg_ses1 = dementia_df[dementia_df['CDR'] == 0.0]['SES'].mean() # Filters where CDR is 0 and finds the mean of SES ''
#print(dementia_df.groupby(by=["CDR", "M/F"]).agg({"eTIV": "min", "nWBV": "mean"})) # Prints frame grouped by CDR and Gender columns only and aggregates Min eTIV and Mean nWBV


# get the numerical values
#summary_stats = dementia_df.groupby(by=["CDR", "M/F"]).agg(
#    {"ID": "count", "Age": ["mean", "std"], "Educ": ["mean","std"],
#     "SES": ["mean", "std"], "MMSE": ["mean","std"]}).round(2)

'''Automating Plots'''
#columns_to_plot = ["Age", "Educ", "SES", "MMSE"]

#plt.figure(figsize=(10,7), facecolor="white")

#for (i, colname) in enumerate(columns_to_plot):
#    plt.subplot(2,2,i+1)
#    sns.barplot(
#        x="CDR", data=dementia_df, y=colname,
#        hue="M/F", ci="sd", color="lightblue")
#    plt.title(f"{colname} (Mean ± SD)")

#plt.tight_layout()
#plt.show()

'''Exercise 2
Seaborn package is really powerful and it allows to create complex plots in a simple way. Let’s look at the sns.relplot() (relational plot) for example. By default, it creates a scatter plot between two variables with a possibility to add additional grouping variables.

Plot the relationship between estimated total intracranial volume ("eTIV") and normalized whole-brain volume (nWBV);
Split plots to separate columns according to the clinical dementia rating ("CDR") using col argument;
Set the color of each point according to the gender ("M/F") using hue argument;
Set the size of a point according to the atlas scaling factor ("ASF") using size column.
'''
#print(dementia_df.dtypes)

#plt.rcParams['figure.facecolor'] = 'white' # set figure background as white

# read in the data
#dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv")

#sns.relplot(
#    data=dementia_df,
#    x="eTIV",
#    y="nWBV",
#    col="CDR",        # split by columns by group
#    hue="M/F",        # color points according to the group
#    size="ASF",       # change the size of a point according to the value
#    sizes=(5, 500), # scale of the points
#    col_wrap=2      # split to two columns
#)
#plt.show()

''' November 21st 2025 '''