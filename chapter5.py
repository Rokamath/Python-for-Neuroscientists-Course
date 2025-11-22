'''I/O'''

'''Working with Text Files'''
'''w:''' # Writing the file #
'''a:''' # Appending the file #
'''r:''' # Reading the file #

#my_string = " How are you doing today, Python?"

#with open("new_file.txt", mode="a") as file:  # A -> Adds new text
#    file.write(my_string)

#with open("new_file.txt", mode="r") as file:
#    output = file.read()

#print(output)

'''Files seperated by lines'''
#with open("new_file.txt", mode="r") as file:
#    output = file.readlines()

#print(output)

'''Exercise 1'''

#write the string to a txt file
#append the file to add another string. Note! If you use method "w" instead of "a" the new string will overwrite the existing data in the file. So be careful when trying to add something to the file.
#read in the file (you have to specify the appropriate method)

neuro = "Neuroscience (or neurobiology) is the scientific study \
of the nervous system."

# write the file
with open('neuroscience.txt', mode="w") as file:
    file.write(neuro)

neuro_cont = " It is a multidisciplinary science that combines \
physiology, anatomy, molecular biology, developmental biology, \
cytology, computer science and mathematical modeling to \
understand the fundamental and emergent properties of neurons \
and neural circuits."

# add new string to the file (append mode)
with open('neuroscience.txt', mode="a") as file:
    file.write(neuro_cont)

# read the file and save the output
with open('neuroscience.txt', mode="r") as file:
    imported = file.read()

#print(imported)

'''Exercise 2. Now imagine that we want to save a list, not a string. This becomes a bit tricky. We want to save a txt file with each value from the list on the new line. This can be possible by writing each value separately in a for loop and adding a "\n" string to each value, which is responsible for creating a new line in the file.'''

'''However, now if we read the file back into the Python using read() method it will save the result as a string, which will look like this:'''


''' "malignant\nmalignant\nbenign\nmalignant\nmalignant\nbenign\nbenign\nbenign" '''
''' And that is clearly not the way we would like to see it. ''' 
''' We can use .readlines() method which will return the list with the strings from each file line ''' 
''' The only problem that all values have the new line sign "\n" at the end (['malignant\n', 'malignant\n', 'benign\n', 'malignant\n', 'malignant\n', 'benign\n', 'benign\n', 'benign\n']). Can you fix it? '''

# string with outcomes
#outcomes =  ['malignant', 'malignant', 'benign', 'malignant',
#             'malignant', 'benign', 'benign', 'benign']

# saving the string to txt file
#with open('outcome.txt', mode='w') as file:
#    for val in outcomes:       # write each value from the list at a new line
#        file.write(val + "\n") # adding "\n" creates a new line in a file
        
# import the file as a string
#with open('outcome.txt', mode="r") as file:
#    outcomes_str = file.read()

# import the file as a list
#with open('outcome.txt', mode='r') as file:
#    outcomes_list = file.readlines()

#print("Imported string:")
#print(outcomes_str)

#print("Imported list:")
#print(outcomes_list)

#outcomes_list = list(map(lambda x: x.replace("\n", ""), outcomes_list)) # clean the values in a list
#print("\nFixed list:")
#print(outcomes_list)

''' Working with CSV/Excel files'''
import numpy as np
frmi_smpl = np.loadtxt(fname="C:\Users\space\Documents\Python for Neuroscientists Course\exercises\data\fmri_data.csv", delimiter=";")
print(frmi_smpl.shape)
