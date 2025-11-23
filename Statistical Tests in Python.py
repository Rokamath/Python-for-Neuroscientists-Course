'''Statistics in Python'''

# Scipy.stats module 
# Pingouin

'''It consists of functions to work with the 
    variety of continuous/discrete random variables (such as Normal distribution),
    functions for summary statistics (such as skewness)
    functions to perform statistical tests (such as Shapiro-Wilk test for normality)'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pingouin as pg
from scipy import stats
import numpy as np

'''Population Distributions'''

# set population parameters
pop_mean = 100 # mean
pop_std = 15   # standard deviation

# create an array of 300 numbers from 50 to 150
# equally separated from each other
#x = np.linspace(start=50, stop=150, num=300)
# find values of probability density function that correspond to the x values
#population = stats.norm.pdf(x=x, loc=pop_mean, scale=pop_std)

#plt.figure(figsize=(10,5))
#sns.lineplot(x=x, y=population) # PDF plot
#plt.axvline(
#    x=pop_mean, color="red",
#    linewidth=1, linestyle="dashed",
#    label="mean")
#plt.xlabel("X")
#plt.title("Population PDF", fontsize=18)
#plt.show()

'''What is the value of survival function at point 120? 
Or in other words, what is the probability that random variable X will be greater than 120?'''

# Calculating probabilites #

#prob_x = stats.norm.sf(x=120, loc=pop_mean, scale=pop_std) # x = value threshold, scale is SD
#print(f"P(X>120) = {prob_x:.2f}") # Prints out probability

'''Sample Distribution'''

#sample_n = 100
# draw a sample of n values from a population
#sample = stats.norm.rvs(loc=pop_mean, scale=pop_std, size=sample_n, random_state=1)

#plt.figure(figsize=(10,5))
#sns.histplot(sample, bins=15)
#plt.xlabel("X")
#plt.title("Histogram of sample distribution", fontsize=18)
#plt.show()

''' Distributions practice'''

'''Exercise 1.
SSRI (Selective Serotonin Reuptake Inhibitor) is one of the most prescribed classes of antidepressants (for example, Prozac) for patients with major depressive disorder (MDD). 
However, studies suggest that around 38% of patients have experienced at least one side effect (like sexual dysfunction, sleepiness or weight gain). 
You have prescribed SSRI antidepressants to 65 new patients.

What is the probability that 25 or more will experience at least one side effect?
In fact, 20 patients of 65 have experienced side effects. What is the probability of such an event?
Plot the probability mass function (PMF) of the random variable X, where X is the number of patients who can experience at least one side effect. '''

''' Binomial distribution'''

#p = 0.38
#n = 65

# 1. Probability that X>=25
#k = 25
#p_x25 = stats.binom.sf(k=k-1, n=n, p=p)
#print(f"P(X>=25)={p_x25:.3f}")

# 2. Probability that X=18
#k = 18
#p_x18 = stats.binom.sf(k=k, n=n, p=p)
#print(f"P(X=18)={p_x18:.3f}")

# 3. Probability Mass Function
#x = np.arange(0,n+1) # sample space (0, 1, ..., n) # cycle through the whole sample
#pmf = stats.binom.pmf(k=x, n=n, p=p)

#plt.figure(figsize=(10,5), facecolor="white")
#plt.bar(x=x, height=pmf, color="lightblue")
#plt.xticks(rotation=-45)
#plt.title("PMF", fontsize=18)
#plt.xlabel("Amount of patients with the side effect")
#plt.show()

'''Exercise 2. t distribution
You have performed a learning task on 20 animals and obtained accuracy values. 
You believe that the animals' performance is below the chance level (50%), so you want to run a t-test to check your hypothesis.

Find 
(1) t score
(2) p-value
(3) t critical (the highest value of t distribution where you would still be able to reject to null hypothesis) using randomly generated values (in a [0, 1] range).'''


#np.random.seed(1) # seed for reproducibility

#alpha = 0.05     # significance level
#null_mean = 0.5 # value under the null hypothesis
#n = 20         # sample size

#sample = np.random.rand(n).round(2) # random values in a range [0, 1]
#sample_mean = sample.mean()
#sample_std = sample.std(ddof=1) # standard deviation
#se = sample_std / np.sqrt(n)                          # standard error

#t_score = (sample_mean - null_mean) / se
#p_val = 1 - stats.t.sf(t_score, df=n-1)
#t_crit = stats.t.ppf(alpha, df=n-1) # threshold value

#x = np.linspace(start=-4, stop=4, num=300) # values for the plot
#t_dist = stats.t.pdf(x, df=n-1)            # pdf

#print(f"t score={t_score:.2f}, p-val={p_val:.2f}")

#plt.figure(figsize=(10,5), facecolor="white")
#plt.plot(x, t_dist, color="black", linewidth=3, label="Null distribution")
# vertical lines with t score and t critical
#plt.axvline(x=t_score, color="blue", label="Observed value")
#plt.axvline(x=t_crit, color="red", label="Threshold value")
# shade area under the curve
#plt.fill_between(x[x<=t_crit], t_dist[x<=t_crit], color='red', alpha=0.5,
#                 label="Rejection area (alpha)")
#plt.fill_between(x[x<=t_score], t_dist[x<=t_score], color='blue', alpha=0.5,
#                 label="p-value")
#plt.xlabel("t")
#plt.ylabel("Density")
#plt.legend()
#plt.show()

'''T-tests'''

'''Now let's perform the t-test on the same data but in more convenient way using ttest_1samp() function from scipy.stats module and ttest() function from pingouin package.'''

#from scipy.stats import ttest_1samp # import just one function
#from pingouin import ttest    # import just one function

#np.random.seed(1) # seed for reproducibility

#null_mean = 0.5 # value under the null hypothesis
#n = 20          # sample size
#sample = np.random.rand(n).round(2) # random values in a range [0, 1]

# scipy implementation
#print("==scipy.stats implementation==")
#t_score, p_val = ttest_1samp(a=sample, popmean=null_mean, alternative="less")
#print(f"t score={t_score:.3f}, p-val={p_val:.3f}")

# pingouin implementation
#print("\n==pingouin implementation==")
#result = ttest(x=sample, y=null_mean, alternative="less")
#print(result.round(3))

'''ANOVA'''

#Is there difference in the average normalized whole-brain volume ("nWBW") among patients with different clinical dementia rating ("CDR")?

'''Null hypothesis: there are no differences between average values of nWBW among groups;
Alternative hypothesis: there is a difference and at least one pair is significantly different;
Significance level: alpha = 0.05.'''

'''ANOVA is a simple F test that measures a ratio:'''

# F = \frac{\text{Explained Variability}}{\text{Unexplained Variability}}
# df_{\text{total}} = n - 1
# df_{\text{group}} = k -1
# df_{\text{error}} = df_{\text{total}} - df_{\text{group}}
# n - total number of observations;
# k - number of groups.

'''Your task:

Read in the data with dementia cases (path to file: "exercises/data/oasis_cross-sectional.csv");
Make a boxplot with CDR levels on x axis and nWBW on the y axis to check the distribution. 
    You will see that there are just two observations with CDR = 2. 
        Filter them out from the DataFrame so they don't affect the result.
Perform the ANOVA and establish the p-value.

There is a function in scipy.stats module f_oneway() which performs one-way ANOVA. 
However, the result is just two objects: test statistic (F) and p-value. 
    Alternative to that, there is an anova() function (from pingouin package) 
    which returns a much more detailed outcome. 
Include them all to compare the output and make a decision about the test.'''

# read in the data
dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv")

# create a boxplot
'''plt.figure(figsize=(9,5), facecolor="white")
sns.boxplot(
    x="CDR", y="nWBV",
    data=dementia_df, palette="vlag",
    width=.5, showmeans=True)
# add points to the plot
sns.stripplot(
    x="CDR", y="nWBV",
    data=dementia_df, size=4,
    color=".3", linewidth=0)
#plt.title(
    "Distribution of eTIV",
    fontsize=18)
plt.show()'''

# scipy.stats implementation:
F_stat, p_val = stats.f_oneway(
    dementia_df["nWBV"][dementia_df["CDR"] == 0],
    dementia_df["nWBV"][dementia_df["CDR"] == 0.5],
    dementia_df["nWBV"][dementia_df["CDR"] == 1])

#print("==scipy.stats implementation==")
#print(f"Calculated test statistic: {F_stat: .3f}\np-value: {p_val: .3f}")

# pingouin implementation:
result = pg.anova(
    dv='nWBV', # dependent variable
    between='CDR', # between-subject factor
    data=dementia_df[dementia_df["CDR"] != 2], # don't forget to filter the df
    detailed=True)

#print("\n==pingouin implementation==")
#print(result)

'''Pairwise t-tests'''

'''
Run pairwise comparison. The easy way to do this is to run pairwise_ttests() function from pingouin package;
Set effect size to Cohen d and adjust the p-value using Bonferroni correction;
Make a conclusions.'''

# Note that Cohen's d value being > 0.5 means medium effect size

# read in the data
#dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv")

result = pg.pairwise_tests(
    dv='nWBV',          # dependent variable
    between='CDR',     # between-subject factor
    data= dementia_df[dementia_df["CDR"] != 2],       # Filters df and where the variable 'CDR' has only 2 observations
    padjust="bonf",  # Bonferroni correction of pvalues
    effsize="cohen"  # include Cohen d effect size
)

print(result.round(2))

'''Chi-squared tests'''
# Chi-squared goodness of fit test #

''' Does clinical dementia rating (CDR) depend on level of education (Educ)? '''

'''Null hypothesis: clinical dementia rating and level of education are independent of each other;
Alternative hypothesis: clinical dementia rating and level of education are dependent (clinical dementia rating varies by level of education);
Significance level: alpha = 0.05.'''

#x**2=sum(k)*((O-E)**2)/E
#df = (R-1)(C-1)

'''
O: observed data in a “cell”
E: expected data of a “cell”
k: number of “cells”
R: number of rows
C: number of columns'''

'''
Read in the data with dementia cases (path to file: "exercises/data/oasis_cross-sectional.csv");
Create a cross table of counts of observations between education level ("Educ") and clinical dementia rating ("CDR"). This can be done using pd.crosstab() function;
Pass the resulting cross table into stats.chi2_contingency() function. The function returns three values: test statistic, p-value of the test, degrees of freedom and expected frequencies;
Find the critical value of Chi-squared for these degrees of freedom and desired significance level;
Create values x and y to plot the null distribution;
Make a plot;
Fix the if statement to make a correct output.'''

# read in the data
#dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv")

# cross table with count of observation for each CDR&Educ combination
#ct = pd.crosstab(
#    index=dementia_df["Educ"],
#    columns=dementia_df["CDR"])

#print("Cross table:")
#print(ct)

# extract the test results
#chisq_stat, p_val, dof, expctd = stats.chi2_contingency(
#   observed=ct)

#alpha = 0.05 # significance level
# percent point function (P[Chi-squared]<q)
# to find the critical value threshold you need to find such value X which has a following property: P(χ²<X) = 1-α or P(χ²>X) = α;
#threshold = stats.chi2.ppf(q=1-alpha, df=dof) # chi-squared critical

# values for the PDF plot
#x = np.linspace(0, 30, 1000)
#y = stats.chi2.pdf(x=x, df=dof) # probability density function

#plt.figure(figsize=(9,5), facecolor="white")
#plt.plot(x, y, color="black", linewidth=3, label="Null distribution")
#plt.axvline(x=chisq_stat, color="blue", label="Observed value")
#plt.axvline(x=threshold, color="red", label="Threshold value")
#plt.fill_between(
#    x[x>=threshold], y[x>=threshold],
#    color='red', alpha=0.5,
#    label="Rejection area (alpha)")
#plt.fill_between(
#    x[x>=chisq_stat], y[x>=chisq_stat],
#    color='blue', alpha=0.5, label="p-value")
#plt.legend()
#plt.xlabel("Chi squared")
#plt.ylabel("Density")
#plt.title("Null Distribution")
#plt.show()

#print(f"Chi-squared={chisq_stat:.2f}, p-val={p_val:.2f}")
#if p_val < alpha:
#    print("Reject the H_0 in favor of H_A")
#else:
#    print("Fail to reject the H0")

'''Confidence intervals for Mean and Bootstrapping'''

# In order to estimate the confidence interval (CI) for a continuous variable you can do it using t distribution for example:
#    \left( \bar{x} - t_{\alpha/2,df} \times \text{SE}; \bar{x} + t_{\alpha/2,df} \times \text{SE} \right )
# \text{SE} = \frac{s}{\sqrt{n}}
# Where SE is a standard error of the mean, alpha is a significance level and t is a quantile of a t distribution.

#Quite often CIs are estimated using bootstrapping technique (especially when the sample size is small). 
#Bootstrapping is any test or metric that uses random sampling with replacement (e.g. mimicking the sampling process), and falls under the broader class of resampling methods. 
# Bootstrapping assigns measures of accuracy (bias, variance, confidence intervals, prediction error, etc.) to sample estimates. 
# This technique allows estimation of the sampling distribution of almost any statistic using random sampling methods.

#The easiest way to perform a bootstrap CI is with the help of compute_bootci() function from pingouin package. 
#The function allows estimating the CI for different statistics (such as Pearson correlation for bivariate data or standard deviation for univariate data) which is 
#    specified by func argument and 
#    returns a numpy array with lower and upper bounds.


'''Your task:

Read in the data with dementia cases (path to file: "exercises/data/oasis_cross-sectional.csv");
Calculate a 95% CI for the average values of normalized whole brain volume (nWBW) for each level of clinical dementia rating (CDR) 
and save the results in a dictionary in the following way:'''

#{0: {'CI': array([0.7615, 0.777 ]), 'mean': 0.7692},
# 0.5: {'CI': array([0.7213, 0.7376]), 'mean': 0.7294},
# 1: {'CI': array([0.6952, 0.7177]), 'mean': 0.7062}}

import pprint
from pingouin import compute_bootci

# read in the file
#dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv") 

# create an empty dictionary
nwbv_estimation = dict()

for cdr_level in [0, 0.5, 1]: # iterate over possible CDR values
    # average values of a sample
    x_bar = dementia_df['nWBV'][dementia_df['CDR'] == cdr_level].mean()
    # add to the dictionary
    nwbv_estimation[cdr_level] = {"mean": round(x_bar, 4)}
    ci = compute_bootci(
        x=dementia_df['nWBV'][dementia_df['CDR'] == cdr_level], # sample
        func='mean',     # CI for the mean
        method='norm',   # normal approximation method
        confidence=0.95, # confidence level
        decimals=4)      # number of rounded decimals to return
    # update the dictinary value
    nwbv_estimation[cdr_level].update({"CI": ci})

#pprint.pprint(nwbv_estimation)

'''Outliers'''

'''How to detect'''

'''Any value that is two or more standard deviation away from the mean (only for normal distribution);
Any value that is out of range of 5th and 95th percentile;
Any value that is beyond the range of [Q1 - 1.5 IQR; Q3 + 1.5 IQR], where Q1 - first quartile, Q3 - third quartile, IQR - interquartile range (Q3-Q1).

The easiest way is to detect the outlier visually is through a histogram or a boxplot.'''

'''Your task is:

Read in the data with dementia cases (path to file: "exercises/data/oasis_cross-sectional.csv");
Calculate the threshold values (save as lower_bound and upper_bound variables) for outliers detection using mean ± 2std method in the atlas scaling factor variable ("ASF");
Make two plots, a histogram with vertical lines for threshold values and a boxplot;
Filter those observations that are beyond the threshold values that you calculated.'''

# read in the file
dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv")

# thresholds for outliers detection (mu ± 2*sigma (ST deviation))
lower_bound = dementia_df['ASF'].mean() - 2*dementia_df['ASF'].std(ddof=1)
upper_bound = dementia_df['ASF'].mean() + 2*dementia_df['ASF'].std(ddof=1)

plt.figure(figsize=(10,6), facecolor="white")
plt.subplot(211)
# histogram
sns.histplot(x=dementia_df['ASF'], color="lightblue", linewidth=2)
# add lines with thresholds
plt.axvline(x=lower_bound, linestyle="dashed",color="red", label="$\mu ±2 \sigma$")
plt.axvline(x=upper_bound, linestyle="dashed", color="red")
plt.xticks([])
plt.xlabel("")
plt.title("Distribution of atlas scaling factor (ASF)", fontsize=18)
plt.legend()

plt.subplot(212)
# boxplot
sns.boxplot(x=dementia_df['ASF'], color="lightblue", width=0.5, linewidth=2)
plt.legend()
plt.show()

# filter those observations that are beyond the threshold values
condition = (dementia_df['ASF'] < lower_bound) | (dementia_df['ASF'] > upper_bound)
outliers_df = dementia_df[condition]
print("\nObservations with extreme ASF values:")
print(outliers_df)


'''Note that mean±2std method resulted in a larger amount of possible outliers (19) compared to 1.5IQR method (2).'''