# Python for Neuroscientists course #
# November 1st 2025 #
# 1 - 5 

# print("Hello World")

x = 4
# y = x ** 0.5
# print(y)

## Question 2 ##
## An aqueous 0.300 M glucose solution is prepared with a total volume of 0.150L. 
# The molecular weight of glucose is 180.16 g/mol. What mass of glucose (in grams) is needed for the solution? ##

c = 0.3 # M
v = 0.15 # L
mr = 180.16 # g/mol

# n = cv
# m = n * mr
n = c * v * mr

# Print the mass in grams to 2 dp
# print (round(n,2))


# Operation	Note
# x | y	## if x is False, then y, else x ##
# x & y	## if x is False, then x, else y ##
# not x	## if x is False, then True, else False ##

# Wanting a male who is above 45
# (sex == "Male") & (age > 45)

# Comparisons 

y = 20

# OR function # = (x) | (y)
# result =(x > 10) | (y < 5)
# result =(x > 10) & (y < 5)
# print(result)
# print(type(result))

# 6 Filtering Out Participants 

# Exercise 1. You run an experiment with two groups: control ("control") and treatment ("treatment"). You want to filter out some participants from the treatment group who don’t meet the minimum BMI criteria (BMI should be equal to or greater than 15). Does this participant meet this criterion? #

age = 30
group = "control"
BMI = 20

#condition = (group == "treatment") & (BMI >= 15)
# print(condition)

# Exercise 2. Now you want to be more sophisticated (for whatever reason). You update your criteria for the treatment group. You want to keep the participant if he is older than 40 OR his BMI equals or greater than 15. Does this participant fit the updated conditions?

condition = (group =="treatment") & ((age > 40) | (BMI >= 15))
# print(condition)

# November 5th 2025 
# 7 Working with Strings 

x = "Neuroscience rocks!"
# print the length of the string x #
# print(len(x))

# print("You " + "can " + "also " + "add " + "strings!") # Adds different strings together
# print("Or even multiply! "*3) # Multiplies the strings by 3

# Slicing 
# You can get slices from the string #

# string [start:end:step] # start index INCLUDED, end index EXCLUDED, step of slicing
#print(x[0:5:1]) # PRINTS # Neuro
#print(x[5:13:1]) # PRINTS # science
#print(x[5]) # PRINTS # Nscc
#print(x[:5]) # PRINTS # Neuro
#print(x[::2]) # PRINTS #Nuocec ok! # Every 2nd value
#print(x[::-1]) # PRINTS # !skcor ecneicsorueN # REVERSE of string

# Methods of analysing Strings 

# basic string methods (does not modify the original string)
x.lower()                  # returns 'neuroscience rocks!'
uc = x.upper()                  # returns 'NEUROSCIENCE ROCKS!'
x.startswith('brain')      # returns False
x.endswith('!')            # returns True
x.isdigit()                # returns False
x.find('science')          # returns index of first occurrence, which is 5
x.find('Psychology')       # returns -1 since not found
x.replace('Neuro','Brain') # replaces all instances of 'Neuro' with 'Brain'

# print(uc)
# print(x)

# Check Occurence of phrase in String #
# Strings are case sensitive # "CUC" =X "cuc"

mRNA = "GUAUGCACGUGACUUUCCUCAUGAGCUGAU"
arginine_codon = "CGC"
#print(arginine_codon not in mRNA)

leucine_codon = "CUC"
#print(leucine_codon in mRNA)

leucine_codon = "cuc"
#print(leucine_codon in mRNA)

# 8 Exercise 1 in Strings 

# Exercise 1. An RNA string is a string formed from the alphabet containing ‘A’, ‘C’, ‘G’, and ‘U’. Given a DNA string corresponding to a coding strand, its transcribed RNA string is formed by replacing all occurrences of ‘T’ with ‘U’. Get the transcribed RNA string in all capital letters.#

DNA = "gatggaacttgactacgtaaatt"
DNAuc = DNA.upper()
RNA = DNA.replace("t", "u").upper()
# print(RNA)
# print(DNAuc)

# Exercise 2 in Strings #
#A palindromic sequence is a nucleic acid sequence in a double-stranded DNA or RNA molecule wherein reading in a certain direction (e.g. 5’ to 3’) on one strand matches the sequence reading in the opposite direction (e.g. 5’ to 3’) on the complementary strand. 
# Is the given sequence a palindromic sequence? #

five_to_three = "GGATCC"
three_to_five = five_to_three.replace("G","c").replace("C","g").replace("A","t").replace("T","a") # replace base with complement in lc
three_to_five=three_to_five.upper()[::-1]  # reverse string as Sequence is in OPPOSITE DIRECTION and make uc
is_palindrome = five_to_three == three_to_five # A logical argument asking where the original string is EXACTLY == the new string
#print(is_palindrome) # Printing answer

## 9 Collections 

# Lists remain the same #
age = [15,18,35,43,23,32]
#print(type(age))

age[:5]   # returns [15,18,35,43,23]
age[4:6]  # returns [23,32]
age[::-1]  # returns [32, 23, 43, 35, 18, 15]
age[-2]    # returns 23
#print(age) # object stayed unchanged
#print(len(age)) # length of list
#print(min(age)) # min of list
#print(max(age)) # max of list
#print(sum(age)) # sum of list

avg_age = sum(age) / len(age) # divides sum of list / length of list
# print(round(avg_age,2)) # average list age rounded to 2 dp

i_am_valid_list = [1, "Hello", [1,2,False], True-0, 42<3.14] # Prints the 3 items // True if 42 < 3.14 
# print(i_am_valid_list)

# Methods == function? 
# No 
#  Think of methods as a function, that could be applied only on a specific data type. Whereas function len() for example can be applied on strings, lists and many other objects. 
# We call function by function(object) and method by object.method(). #

participants = ['Bob', 'Bill', 'Sarah', 'Max', 'Jill']
# methods that modify the initial list
participants.append('Jack') # append one element to the end
# ['Bob', 'Bill', 'Sarah', 'Max', 'Jill', 'Jack']
participants.extend(['Anna', 'Bill']) # append multiple elements to the end
# ['Bob', 'Bill', 'Sarah', 'Max', 'Jill', 'Jack', 'Anna', 'Bill']
participants.insert(0, 'Louis') # insert the element at index 0 (shifts everything to the right)
# ['Louis', 'Bob', 'Bill', 'Sarah', 'Max', 'Jill', 'Jack', 'Anna', 'Bill']
participants.remove('Jill') # searches for the first instance and removes it
# ['Louis', 'Bob', 'Bill', 'Sarah', 'Max', Jack', 'Anna', 'Bill']
participants.pop(1) # removes the element at index 1 and returns it
# ['Louis', 'Bill', 'Sarah', 'Max', Jack', 'Anna', 'Bill']
# methods that don't modify initial list and return a new object
#print(participants.count('Bill')) # returns the number of instances
#print(participants.index('Max'))  # returns the index of the first instance
2
3
# not a method, but in this way you can change the value(s) of the list
participants[2] = 'Ben' # replace the element at the index 2
# ['Louis', Bill', 'Ben', 'Max', Jack', 'Anna', 'Bill']

# Tuples 
# A type of Collection which is ORDERED & UNCHANGABLE 
#print("For Tuples" + ":" + '\033[1m' + " Always" + '\033[0m '+ "add items within" + "()") # How to bold text use # '\033[1m' + "Always" + '\033[0m '# 
brain_lobes = ('frontal', 'parietal', 'temporal', 'occipital')
# or:
# brain_lobes_list = ['frontal', 'parietal', 'temporal', 'occipital']
# brain_lobes = tuple(brain_lobes_list)
#print(type(brain_lobes))

#brain_lobes[0] = 'anterior'

# Sets 
# Unsorted and unindexed collection with no duplicates 

languages = {'python', 'r', 'java'} # create a set directly
snakes = set(['cobra', 'viper', 'python']) # create a set from a list

#Set operations:
languages & snakes # intersection, AND
languages | snakes # union, OR // . This fact can become handy used when looking for the unique values in a list.
#print(languages - snakes) # set difference

# Dictionaries 
# Unordered, Iterable, Mutable
# Dictionaries need keys to access data and thus dictionaries don't allow duplicated keys

participant = {'name': 'Jon Doe', 'group': 'Control', 'age': 42}
(participant['name'])

# add new key-value pair to the dictionary
participant['ID'] = 'CJD'
#print(participant)

#print(participant.keys()) # shows labels in dictionary

#print(participant.values()) # shows values in dictionary for all


# 10 Exercises
# Exercise 1. You performed EEG recordings from a subject in 10 trials total. However, some trials have been marked as “BAD” due to eye blinks and bad electrodes’ connection. Exclude bad trials from the total list of trials and save it as a new object. good_trials should be in a list format.#

trial_ids = ["001", "002", "003", "004", "005",
             "006", "007", "008", "009", "010"]
bad_trials = ["004", "006", "007"]

trial_ids = set(trial_ids) # Convert list to set
bad_trials = set(bad_trials) # Convert list to set
good_trials = trial_ids - bad_trials # Subtract set difference
good_trials = list(good_trials)
#print(good_trials)

# Exercise 2. Count how many times adenine (A), cytosine (C), guanine (G), and thymine (T) nucleotides appear in the given DNA string. Save result as a dictionary with four keys, where keys represent the nucleotide and values represent the counts. You can use either the first letters for the dictionary keys (A) or the full names (Adenine).

DNA = "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC"

A = (DNA.count('A')) 
G = (DNA.count('G')) 
C = (DNA.count('C')) 
T = (DNA.count('T')) 

nucleotides = {'A':A,'G':G,'C':C,'T':T} # creates LABELS in dictionary per BASE and adds VALUES of COUNT VALUES
print(nucleotides) # prints dictionary
