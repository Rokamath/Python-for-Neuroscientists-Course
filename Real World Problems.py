import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import pingouin as pg
from scipy.io import loadmat
from sklearn import tree 
from scipy.special import softmax

'''Real World Problems'''

'''Exercise from the textbook Theoretical Neuroscience by Dayan and Abbott (Chapter 1, exercise 8):

H1_neuron.mat file contains data collected and provided by Rob de Ruyter van Steveninck from a fly H1 neuron responding to an approximate white-noise visual motion stimulus. 
Data were collected for 20 minutes at a sampling rate of 500 Hz. 
In the file, rho is a vector that gives the sequence of spiking events or nonevents at the sampled times (every 2 ms). 
When an element of rho is one, this indicates the presence of a spike at the corresponding time, whereas a zero value indicates no spike. The variable stim gives the sequence of stimulus values at the sampled times. 

Calculate and plot the spike-triggered average from these data over the range from 0 to 300 ms (150 time steps).

The spike-triggered average (STA) is a tool for characterizing the response properties of a neuron using the spikes emitted in response to a time-varying stimulus. The STA provides an estimate of a neuron’s linear receptive field.'''

# import the file
#h1_data =loadmat(file_name="exercises/data/H1_neuron.mat",
#    squeeze_me=True # squeeze the file to remove empty indexes
#    )

# create a new key with the time points
# from 0 to the length of the data
#h1_data["timepnt"] = np.arange(0, len(h1_data['rho']), 1)
# select only those time point when spike occurred
#h1_data["spike_time"] = h1_data["timepnt"][h1_data["rho"]  == 1]
# set the window size (timepoints)
#window = 150

# create a vector of zeros with the shape (window,)
#h1_data["sta"] =  np.zeros(shape=(window,))
# iterate over all timepoints when spike occurred
# and use the time point as a end index
# note that we cannot take the window of length 150 for first 149 observations
#for end_index in h1_data["spike_time"][h1_data["spike_time"]>window]:
    # specify the start index of a window
#    start_index = end_index - window
    # take the slice from the stimulus value
#    sample = h1_data["stim"][start_index:end_index]
    # add the slice to the STA vector
#    h1_data["sta"] += sample

# divide the resulting STA vector on the amount of time points
# to get the actual average
#h1_data["sta"] /= len(h1_data["spike_time"][h1_data["spike_time"]>window])

#plt.figure(figsize=(10,5), facecolor="white")
#plt.plot(range(0, 300, 2), h1_data["sta"])
#plt.title('Spike-Triggered average', fontsize=18)
#plt.ylabel('Stimulus Value, [mV]')
#plt.xlabel('Time, [ms]')
#plt.show()

'''Reducing uncertainty using Decision Tree Modelling'''

'''Decision tree learning is one of the predictive modeling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item’s target value (represented in the leaves). Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Decision trees are among the most popular machine learning algorithms given their intelligibility and simplicity.

Classification and regression trees (CART for short) models are not the first-choice models when it comes to prediction because they tend to overfit the data (in other words they can predict quite good on the training data, but much worse on the test data), but they are really good in explaining the data structure. If you are interested in the math behind the model, check out the links below. However, you don’t need to know much to complete the exercise.'''

'''Exercise. What features can be associated with the dementia rating?

Read in the data with dementia cases (path to file: "exercises/data/oasis_cross-sectional.csv");
Drop the redundant columns: "ID" (ID label shouldn’t be a predictor of Alzheimer’s, should it?), "Hand" (all observations are right-handed) and "Delay" (most of the values are missing) and save it in the new data frame model_data (even though the columns might be meaningless, it’s a good idea to keep the raw data);
Drop the rows with missing values. Note, this is a sloppy solution for a missing data problem. There are several methods on how you can artificially replace the missing values and some new algorithms can handle missing data during modeling, however, this is beyond this exercise;
We are going to build a binary classification tree: 0 for no dementia (CDR is 0) and 1 for dementia status (CDR is 0.5, 1 or 2). Create new binary column dementia and drop the "CDR" column;
scikit-learn models don’t allow string columns (unlike models in R), that’s why we often have to perform some sort of feature engineering. In our case we just need to convert gender column M/F to numerical binary - 1 if female, 0 otherwise;
Split the model_data into data frame with independent variables (all features, X) and a series with the dependent variable (binary, y) for later use;
Build a model using entropy for split criteria and set the maximum depth of the tree to 3;
Fit the data to the model and make a plot.'''

# import the file
#dementia_df = pd.read_csv("exercises/data/oasis_cross-sectional.csv")

# drop redundant columns
#model_data = dementia_df.drop(columns=["ID", "Hand", "Delay"])
#model_data.dropna(inplace=True) # drop missing values
# create a new binary column, 1 if subject has dementia, 0 otherwise
#model_data["dementia"] = model_data["CDR"].apply(lambda x: 1 if x > 0 else 0)
# drop CDR column
#model_data.drop(columns="CDR",inplace=True)
# convert gender column to numerical, 1 if female, 0 otherwise
#model_data["M/F"] = model_data["M/F"].apply(lambda x: 1 if x == "F" else 0)

#X = model_data.drop(columns="dementia") # dataframe with features only
#y = model_data["dementia"] # column with the dementia status

# build the model
#model = tree.DecisionTreeClassifier(
#    criterion='entropy',
#    max_depth=3)

# fit the data
#model.fit(X, y)

#plt.figure(figsize=(11, 7), facecolor="white")
#tree.plot_tree(
#    decision_tree=model,
#    filled=True, # fill the cells according to the class
#    feature_names=X.columns.tolist(),
    # use proportions of class occurrence instead of absolute values
#    proportion=True,
#    class_names=['No Dementia', 'Dementia'])
#plt.show()

'''The very first node is the initial data (X and y). IF True LEFT or False Right'''
'''At each node, you can see the entropy in the y variable, fraction of observations from total and ratio of two classes.'''

'''Simulating Reinforcement Learning'''

'''The agent interacts with the environment by making the actions. 
    Each action results in the outcome reward and brings the agent to a new state. 
This is a continuous loop and throughout this interaction, 
    agent learns what actions lead to the highest reward by updating his beliefs about the action-state pairs.'''

'''Imagine three slot machines, where each machine has a unique and unknown (to the agent) probability of reward (a.k.a. Three-armed bandit task).

Your task is to create the agent, who has to learn which option is more rewarding and maximize the total reward within the limited number of trials. For this, you are going to use the Q-learning model, which is most commonly used for decision-making modeling. There are several modifications of this model with different parameters, but in a simple form algorithm can be represented like this:'''

'''α is a learning rate, which lies in a range [0, 1] and controls the speed of updating the value function Q. The lower the value, the slower the learning process.
β is an inverse temperature, with the lower non-inclusive bound 0. Controls the behavior of the agent - the lower the value, the more explorative the agent.
Q(a) is a value function or expected reward of an action a.
The softmax function transforms the value function values to probabilities of making an action. It does that by computing the exponential of each value divided by the sum of the exponentials of all the values, scaled by the inverse temperature. In Python it can be calculated using softmax() function from scipy.special module.'''


# specify random generator
rnd_generator = np.random.default_rng(seed=123)
# colors for the plot
colors_opt = ['#82B223', '#2EA8D5', '#F5AF3D']

n_arms = 3    # number of arms (slot machines)
opts = list(range(n_arms)) # option numbers (0, 1, ..., n_arms)
n_trials = 100  # number of trials
alpha = 0.3     # learning rate
beta = 2        # inverse temperature
rew_prob = [0.2, 0.4, 0.8]  # probability of reward for each arm

# arrays that will hold historic values
# selected option at each trial
actions = np.zeros(shape=(n_trials,), dtype=np.int32)
# observed reward at each trial
rewards = np.zeros(shape=(n_trials,), dtype=np.int32)
# value function for each option at each trial
Qs = np.zeros(shape=(n_trials+1, n_arms))
# note that before the first trial agent has already expectations of each
# option (0s in our case). That means that on the trial t we are going to
# update the Q for the (t+1) trial. To update the value function on the last
# trial we include `+1` in the Q array shape

for i in range(n_trials): # loop over all trials

    # choose the action based of softmax function
    prob_a = softmax(beta * Qs[i, :]) # probability of selection of each arm
    a = rnd_generator.choice(a=opts, p=prob_a)  # select the option
    # list of actions that were not selected
    a_left = opts.copy()
    a_left.remove(a) # remove the selected option
    # check if arm brigns reward
    if rnd_generator.random() < rew_prob[a]:
        r = 1
    else:
        r = 0
    # value function update for the chosen arm
    Qs[i+1, a] = Qs[i, a] + alpha * (r - Qs[i,a])
    # update the values for non chosen arms
    for a_l in a_left:
        Qs[i+1,a_l] = Qs[i,a_l]
    # save the records
    actions[i] = a
    rewards[i] = r

# calculate cumulative reward
cum_rew = rewards.cumsum()
# count how many times each arm was chosen
unique, counts = np.unique(actions, return_counts=True)

plt.figure(figsize=(10,5), facecolor="white", )

plt.subplot(221)
for i in range(n_arms):
    plt.plot(
        Qs[:, i],
        color=colors_opt[i],
        lw=2,  #line width
        label=f'Arm #{i+1}'
    )
plt.xlabel('Trial')
plt.ylabel('Value Function')
plt.legend()

plt.subplot(223)
plt.plot(cum_rew, color='black')
plt.xlabel('Trial')
plt.ylabel('Cumulative Reward')

plt.subplot(122)
plt.bar((unique + 1).astype(str), counts, color=colors_opt)
plt.xlabel('Arm Number')
plt.ylabel('# of Times Chosen')

plt.suptitle('Agent\'s Performance', fontweight='bold')
plt.tight_layout()
plt.show()