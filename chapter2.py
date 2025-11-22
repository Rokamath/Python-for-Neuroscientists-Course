# November 17th 2025

import matplotlib.pyplot as plt

''' IF Statements '''

''' if condition is True:
#    do something
# elif another condition is True:
#    do something
# else:
#    do something else
'''

a = 100
b = 500

# if a > b:
    # print('A is greater than B')
# elif a == b:
    # print('A equals B')
# else:
    # print('A is smaller than B')

''' WHILE Loops  execute the set of statements as long as condition is True

 while condition is True:
   do something '''

# x = 0

# while x < 4:
   # x += 1 # which is the equivalent of `x = x + 1`
   # print(x)

# FOR Loops iterating over a sequence (like lists, tuples, dictionaries, sets, strings).

# Word List
my_list = ['data', 'science', 'rocks']

# for word in my_list:     # iterate over all values in the list
    # print(word.upper())  # change to upper register and print out the value

# Number Data
# stdev = [2, 4, 1.5, 2, 4]     # list of standard deviations

#variances = []                # initialize the empty list that will hold variances
# for val in stdev:             # iterate over all values in the list
#    variances.append(val**2)  # append the variances list with squared value

# print(variances)

''' If you need to break the loop before it has looped through all the items (in the for loop) or even if the statement is still True (in the while loop) you can use the "break" statement.
 With the "continue" statement you can stop the current iteration of the loop and continue with the next one. '''

# Starts a New step of a loop using "continue"

#stdev = [2, 4, 0, 1.5, 2, 4]
#variances = []
#for val in stdev: # example 1
#    if val == 0:
#        continue  # exit the current step of the loop
#    variances.append(val**2)
# print(variances)

# Break
# variances2 = []
# for val in stdev: # example 2
#    if val == 0:
#        break  # exit the entire loop
#    variances2.append(val**2)
# print(variances2)

# List Comprehensions #

# Original Print
#stdev = [2, 4, 0, 1.5, 2, 4]
#variances = [val**2 for val in stdev]
#print(variances)

# Not printing 0 values
#variances = [val**2 for val in stdev if val != 0]
#print(variances)

# Printing wrong value if 0
#variances = [val**2 if val != 0 else "wrong value" for val in stdev]
#print(variances)

''' Exercise 1. Given a string my_string count all word characters, digits, and special symbols in it. For example, string "h336^" has 
# 1 word character
# 3 digits
# 1 special symbol '''

my_string = "P@#yn26at^&i5ve"

# set the initial values for the counter
word_characters_count = 0
digits_count = 0
symbols_count = 0

# iterate over the string
for char in my_string:
    if char.isalpha():   # check if a word character ('a', 'b', 'c')
        word_characters_count += 1
    elif char.isnumeric(): # check if a digit (1, 2, 3)
        digits_count += 1
    else:      # the rest are symbols ('%', '@', '+')
        symbols_count += 1

# add resulting values to a dictionary
counts = {
    "Word Characters": word_characters_count,
    "Digits": digits_count,
    "Symbols": symbols_count
    }
# print(counts)

''' Exercise 2. Write a code to get the Fibonacci series. The Fibonacci Sequence is the series of numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, .... Every next number is found by adding up the two numbers before it. '''

# Keep just first 10 digits for simplicity.

#fibonacci_series = [0, 1]  # first two values of series

#while len(fibonacci_series) < 10:  # condition for the length
    # add the new value
#    fibonacci_series.append(fibonacci_series[-1]+fibonacci_series[-2])
#print(fibonacci_series)

''' Exercise 3. You performed a learning task and got accuracy (the number of correct trials divided by the total number of trials) for 10 subjects stored in a dictionary. Which 3 subjects performed the best? '''

#accuracy_scores = {
#    'id1': 0.27, 'id2': 0.75, 'id3': 0.61, 'id4': 0.05, 'id5': 0.4,
#    'id6': 0.67, 'id7': 0.69, 'id8': 0.52, 'id9': 0.7, 'id10': 0.3
#    }
# store the top 3 values from the dictionary as a list
#max_accs  = sorted(accuracy_scores.values(), reverse=True)[:3]

#max_keys = sorted(accuracy_scores, key=accuracy_scores.get, reverse=True)[:3]

# create an empty list that will hold ids of participants with the highes accuracy
#max_ids = []    # create an empty list
#for key in max_keys:       # iterate over all keys in the dictionary
#    if key in max_keys:    # check if the value of this key is in top 3
#        max_ids.append(key)          # if so, append the list

#print(max_ids)

''' Exercise 4. The leaky integrate-and-fire (LIF) model is one of the most simplest mathematical model that tried to explain neuron’s behavior. LIF model represents neuron as a parallel combination of a “leaky” resistor with a conductance g_L and a membrane capacitor with a capacitance C_m. If the input current I is sufficiently strong enough such that membrane potential V reaches a certain threshold value V_thresh, V is reset to V_reset. '''

# E_L is the resting potential;
#taum= Cm/gL, membrane time constant.

N = 1000           # number of measurememnts (time points)
g_L = 0.7          # leak conductance [nS]
c_m = 20           # conductance [nF/mm^2]
tau_m = c_m/g_L        # membrane time constant [ms]
i = 50             # input current [nA]
v_reset = -65      # reset potential [mV]
v_thresh = -50     # threshold [mV]
dt = 0.1           # time step
E_L = -75          # leak reversal potential [mV]
v = [v_reset]      # list of potential values, start at v_reset

for _ in range(N): # loop over all time points
    # increment of membrane potential
    dv = (-(v[-1] - E_L)+(i/g_L)) * dt / tau_m
    v_new = v[-1] + dv     # new membrane potential value
    if v_new >= v_thresh:  # check if new value exceeds threshold
        v.append(v_reset)  # add reset value, if true
    else:
        v.append(v_new)  # add new value to the list otherwise

plt.figure(figsize=(10,5), facecolor="white")
plt.plot(v, color='black', lw=4)
plt.axhline(
    v_thresh, linestyle='dashed',
    lw=1, label='Threshold',
    color='red')
plt.axhline(
    v_reset, linestyle='dashed',
    lw=1, label='Reset',
    color='blue')
plt.legend()
plt.xlabel('Time point')
plt.ylabel('V, [mV]')
plt.title('Membrane potential', fontsize=14, fontweight='bold')
#plt.show()

''' Writing Custom Functions '''

''' General form:
 def my_new_function(some_argument1, some_argument2):
    do something
    return something '''

#Example:
def mean(input_list):
    nominator = sum(input_list)
    denominator = len(input_list)
    return nominator/denominator


#l = [1,2,3,4,5]
#print(mean(input_list=l))

''' Function Arguments '''

#def get_pi():
#    pi = 22/7
#    return pi

#print(get_pi())

#def get_pi(circumference, diameter, digits_to_round=2): # rounds result to 2 dp
#    pi = circumference/diameter
#    return round(pi, digits_to_round)

#print(get_pi(circumference=22, diameter=7))
#print(get_pi(circumference=22, diameter=7, digits_to_round=1))

# Map and Filter functions
    ## General Form 
    # map(function, iterable_object)
    # filter(function, iterable_object)

#def squared(x):
#    return x**2

stdev = [1.5,2,2,4,4] # list
# print(list(map(squared, stdev))) # Applies the squared function to every value in the stdev list

def greater_than_two(x):
    return x > 1

result = list(filter(greater_than_two, stdev)) #  Filters the given iterable object with the help of a function that tests each element in the iterable to be TRUE or NOT.

#print(result)

# Lambda Functions
# lambda arguments: expression
    
result1 = list(map(lambda x: x**2, stdev)) # No name function TEMP
result2 = list(filter(lambda x: x>2, stdev)) # No name function TEMP 

''' Exercise 1. Write a function that takes a list of numbers as an input and returns a variance of that list. 

# \bar{x} - the average value,
# N - number of observations. '''

# s**2 = (sum(x - average)**2)/N

#def variance(input_list):           # function with one input argument for a list object
#    N = len(input_list)                  # calculate the sample size
#    x_bar = sum(input_list)/N              # calculate the average value
#    numerator = []         # empty list with the values from the numerator
#    for x in input_list:            # iterate over all values in the given list
#        numerator.append((x - x_bar)**2)   # subtract average value from x, square it, and add to the numerator list
#    return sum(numerator) / N            # return the sum of the numerator divided by N

#participants_height = [167, 185, 179, 191, 178, 180, 175, 188, 170]
#height_var = variance(input_list=participants_height)
# print(height_var)

''' Exercise 2 When we are dealing with the sample variance we have to subtract 1 degree of freedom in the denominator. '''

''' Update your function by adding df argument that will specify how many degrees of freedom we want to subtract.
Set the default value to 1 (meaning that by default it will use the formula for the sample variance).'''

#def variance(input_list):           # function with one input argument for a list object
#    N = len(input_list)                  # calculate the sample size
#    x_bar = sum(input_list)/N  # calculate the average value
#    df = 1            # set degrees of freedom
#    numerator = []         # empty list with the values from the numerator
#    for x in input_list:            # iterate over all values in the given list
#        numerator.append((x - x_bar)**2)   # subtract average value from x, square it, and add to the numerator list
#    return sum(numerator) / (N-df)            # return the sum of the numerator divided by N

#participants_height = [167, 185, 179, 191, 178, 180, 175, 188, 170]
#height_var = variance(input_list=participants_height)
#print(height_var)

''' Exercise 3. You performed a learning task and got accuracy (the number of correct trials divided by the total number of trials) for 10 subjects stored in a dictionary. Now you want to save the IDs of those participants whose score was above 0.6 in a separate list. How could you solve it using filter and lambda function? '''

accuracy_scores = {
    'id1': 0.27, 'id2': 0.75, 'id3': 0.61, 'id4': 0.05, 'id5': 0.4,
    'id6': 0.67, 'id7': 0.69, 'id8': 0.52, 'id9': 0.7, 'id10': 0.3
    }

good_ids = list(filter(lambda x: accuracy_scores[x] > 0.6, accuracy_scores.keys()))
#print(good_ids)

''' Exercise 4. Your experiment had 5 subjects in a control group and 5 subjects in a treatment group. However, the labels were written in a funny way. 
 Clean the groups list, so values are "control" and "treatment" using map and lambda functions. 
 Then using for loop and if statement separate IDs into control and treatment lists. '''

ids = [
    'id1', 'id2', 'id3', 'id4', 'id5',
    'id6', 'id7', 'id8', 'id9', 'id10'
    ]
groups = [
    'ContrOl_1', 'ContrOl_1', 'TreatMent_2', 'ContrOl_1', 'TreatMent_2',
    'TreatMent_2', 'TreatMent_2', 'TreatMent_2', 'ContrOl_1', 'ContrOl_1'
    ]

groups_cleaned = list(map(lambda x: 'control' if 'control' in x.lower() else 'treatment', groups)) # a lambda function which sorts groups into control if that WORD is in the group, else treatment

ids_ctl = []             # initialize empty list with control group IDs
ids_trt = []             # initialize empty list with treatment group IDs
for i in range(len(groups_cleaned)):             # iterate over indeces of the list
    if groups_cleaned[i] == 'control':  # if the i-th elemnt in a cleaned groups list is control
        ids_ctl.append(ids[i])       # then add i-th ID to the control group
    else:
        ids_trt.append(ids[i])       # otherwise add i-th ID to the treatment group

#print(f"Control Group: {ids_ctl}")
#print(f"Treatment Group: {ids_trt}")

''' Assert Function 
assert condition, "Message to add to an error in case the condition isn't true" '''

# def get_pi(circumference, diameter, digits_to_round=2):
#    pi = circumference/diameter
#    return round(pi, digits_to_round)


# use a string as an function input
# print(get_pi(circumference=22, diameter="some number"))

''' There is a way to check your input values using assert statement. 
 It works somewhat similar to if statement. We check a condition. 
 If it is True nothing happens and the function executes according to the script. 
 If the condition if False, then you get an AssertionError with the message you specified, and the function is not executed. '''

#def get_pi(circumference, diameter, digits_to_round=2):

""" Function returns pi value based on the circumference
    and diameter values of a circle.

    Parameters
    ----------
    circumference : int or float
        Circumference of the circle.
    diameter : int or float
        Diameter of the circle.
    digits_to_round : int
        Amount of digits to keep after the coma in a pi value.

    Returns
    ----------
    pi : float
        resulting pi value. """

    #assert type(circumference) in [int, float], \
#    "Check the type of the circumference argument. Should be numeric."
    #assert type(diameter) in [int, float], \
#    "Check the type of the diameter argument. Should be numeric."
    #assert type(digits_to_round) == int, \
#    "Check the type of the digits_to_round argument. Should be an integer."

    #pi = circumference/diameter
    #return round(pi, digits_to_round)

#print(get_pi(circumference=22, diameter="some number"))

''' Exercise. Remember the variance() function you created recently? 
# What if the input list holds any non-numeric object? In this case, function will also fail. 
# Check that all the values in the input lists are numeric (int or float) using assert statement. '''

# Also, degrees of freedom should be a positive integer or 0. Raise an error if that’s not true. #
    
def variance(input_list, df=1):

    assert (type(df)==int) & (df >= 0), "Degrees of freedom should be a positive integer or 0"
    # get the list of booleans that were a result of checking
    # if value from a list is either int or float
    is_type_numeric = map(lambda x: type(x) in [int, float], input_list)
    # if the sum of that list doesn't equal the length of an input list
    # that means, that one of more objects were not numeric
    assert sum(is_type_numeric) == len(input_list), \
    "Some of the values in the list are not numeric"

    N = len(input_list)                # calculate the sample size
    x_bar = sum(input_list) / N        # calculate the average value

    numerator = []                     # empty list with the values from the numerator
    for x in input_list:               # iterate over all values in the given list
        numerator.append((x-x_bar)**2) # subtract average value from x,
                                       # square it, and add to the numerator list
    return sum(numerator) / (N-df)     # return the sum of the numerator divided by (N - df)

""" Function returns the variance of a given lists.

    Parameters
    ----------
    input_list : list
        List of numeric values

    df : int, default is 1
        Degrees of freedom to subtract.
        When `df=0` formula is for the population variance.
        When `df=1` formula is for the sample variance.

    Returns
    ----------
    Variance: float
        resulting variance.
    """

try:
    participants_height = [167, 185, 179, 191, 178, 180, 175, 188, "170"]
    height_var = variance(input_list=participants_height)
except:
    height_var = None

#print(height_var)

''' f-Strings '''

''' Imagine you are running a loop over all values of the list and at each step you want to add each value to a string (for example to print out). You can do this using so-called f-strings in the following way: f"{<variable name>}" 
which will dynamically paste the value of a given variable inside the string. For example: '''

participants = ["id1", "id6", "id8"]

for val in participants:
    # print(f"{val}")

    ''' When you pass the float number as a variable inside the f-string, you can also specify additional formatting style, such as 
rounding, in a way ''' 

# f"{<float>: .<digits>f}" 
''' <digits> is number of decimal places '''

scores = [456.1341, 478.12451, 501.2345]

for val in scores:
    # print(f"{val: .2f}")

    ''' Exercise Given the dictionary where key is a subject ID and value is accuracy for the task, 
print out the sting for each participant using for loop in a way: 
    "Accuracy for subject id1 is 0.27.". 
Round the accuracy to two digits after the decimal point.'''

accuracy_scores = {
    'id1': 0.2665, 'id2': 0.7523, 'id3': 0.6123, 'id4': 0.053, 'id5': 0.389,
    'id6': 0.6732, 'id7': 0.692, 'id8': 0.5184, 'id9': 0.743, 'id10': 0.3111
    }

''' for (key, value) in dictionary_name.items()
    .items goes through a dictionary's items '''

for (key, value) in accuracy_scores.items():
    print(f"Accuracy for subject {key} is {value: .2f}.")

''' Formatting '''

'''Type	Naming Convention	Examples
Function	Use a lowercase word or words. Separate words by underscores to improve readability.	function, my_function
Variable	Use a lowercase single letter, word, or words. Separate words with underscores to improve readability.	x, var, my_variable
Class	Start each word with a capital letter. Do not separate words with underscores. This style is called camel case.	Model, MyClass
Method	Use a lowercase word or words. Separate words with underscores to improve readability.	class_method, method
Constant	Use an uppercase single letter, word, or words. Separate words with underscores to improve readability.	CONSTANT, MY_CONSTANT, MY_LONG_CONSTANT
Module	Use a short, lowercase word or words. Separate words with underscores to improve readability.	module.py, my_module.py
Package	Use a short, lowercase word or words. Do not separate words with underscores.	package, mypackage
Names should be meaningful

Maximum line length
from mypkg import example1, \
    example2, example3
    
Example of breaking before a binary operator:
# Recommended
total = (first_variable
         + second_variable
         - third_variable)

# Not Recommended
total = (first_variable +
         second_variable -
         third_variable)
'''

''' Indentations (1) '''

''' Not recommended '''
# Arguments on first line forbidden when not using vertical alignment.
#foo = long_function_name(var_one, var_two,
#    var_three, var_four)

'''further indentation required as indentation is not distinguishable. '''
#def long_function_name(
#    var_one, var_two, var_three,
#    var_four):
#    print(var_one)

'''Recommended '''
# Aligned with opening delimiter.
#foo = long_function_name(var_one, var_two,
#                         var_three, var_four)

'''add 4 spaces (an extra level of indentation) to distinguish arguments from the rest. '''
#def long_function_name(
#        var_one, var_two, var_three,
#        var_four):
#    print(var_one)

'''hanging indents should add a level. '''
#foo = long_function_name(
#    var_one, var_two,
#    var_three, var_four)

#my_list = [
#    1, 2, 3,
#    4, 5, 6
#    ]

#result = some_function_that_takes_arguments(
#    'a', 'b', 'c',
#    'd', 'e', 'f'
#    )

'''Whitespace around binary operators'''

# Recommended
#y = x**2 + 5
#z = (x+y) * (x-y)

# Not Recommended
#y = x ** 2 + 5
#z = (x + y) * (x - y)

'''However: '''

# Recommended
#def function(default_parameter=5):
#    pass

# Not Recommended
#def function(default_parameter = 5):
#    pass