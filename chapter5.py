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

#neuro = "Neuroscience (or neurobiology) is the scientific study \
#of the nervous system."

# write the file
#with open('neuroscience.txt', mode="w") as file:
#    file.write(neuro)

#neuro_cont = " It is a multidisciplinary science that combines \
#physiology, anatomy, molecular biology, developmental biology, \
#cytology, computer science and mathematical modeling to \
#understand the fundamental and emergent properties of neurons \
#and neural circuits."

# add new string to the file (append mode)
#with open('neuroscience.txt', mode="a") as file:
#    file.write(neuro_cont)

# read the file and save the output
#with open('neuroscience.txt', mode="r") as file:
#    imported = file.read()

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
#frmi_smpl = np.loadtxt(fname="exercises\data\'f'mri_sample.csv",delimiter=",") # Argument specifies how observations are separated in the input file.
#print(frmi_smpl.shape) # R and C

'''Creating CSV files with numpy'''
#np.savetxt(fname="one_subject.csv", X=frmi_smpl[0,:], delimiter=",")

'''Reading CSV files with Pandas'''
import pandas as pd

#oasis_df = pd.read_csv(filepath_or_buffer="exercises\data\oasis_cross-sectional.csv", sep=",") # CSV files
#print(oasis_df.head())

#oasis_df2 = pd.read_excel(io="exercises\data\oasis_cross-sectional.xlsx", sheet_name=0) # Excel files with sheet_name as "Sheet1" OR index [1]
#print(oasis_df2.head())

'''Loading files from the Internet'''
#URL = "https://raw.githubusercontent.com/rklymentiev/py-for-neuro/binder/exercises/data/oasis_cross-sectional.csv"

#dementia_df = pd.read_csv(URL) # allows to load files directly from the internet by the direct URL to the file.
#print(dementia_df) # Prints CSV

'''Opening ZIP files using Pandas'''
# Also file can be compressed in an archive (for example, .zip or .7z)
#dementia_df = pd.read_csv(filepath_or_buffer="exercises/data/oasis_cross-sectional.zip") # ADD type of compression - ,compression="zip" 

'''Writing CSV/Excel files with Pandas'''
#dementia_df.to_csv(path_or_buf="dementia_df.csv", sep=",", index=True) # Create CSV file with index as 1st column
#print(dementia_df)

#dementia_df.to_excel( # Create Excel file
#    excel_writer="dementia_df.xlsx",
#    sheet_name='Main Data',
#    index=False)

'''Exercise. fMRI data set
Load in the data from fMRI experiment. 
Path to file "exercises/data/fmri_data.csv". Note that columns in a file are separated by ;.
Create a new DataFrame parietal_df with observations from the parietal region (df["region"] == "parietal");
Save the resulting DataFrame as an Excel file.'''

# read in the data
#df = pd.read_csv("exercises/data/fmri_data.csv",sep=';')
#print(df.head())

# take only the parietal region
#parietal_df = df[df["region"] == "parietal"]
# save as Excel file
#parietal_df.to_excel("parietal_sample.xlsx")

'''Working with MATLAB files'''

'''MAT-files are binary MATLAB files that store workspace variables.'''

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

#h1_data = loadmat(file_name="exercises/data/H1_neuron.mat", squeeze_me=True) #  squeezing unit matrix dimensions. For example, if MATLAB variable was stored in a shape (5,1,1), setting squeeze_me=True will import it with the shape (5,) to Python.
#print(h1_data)

'''
The resulting object is a dictionary. 
Each key represents a saved variable from MATLAB. 
Also, there are three additional keys with the file info. 
    rho is a vector that gives the sequence of spiking events or nonevents at the sampled times (every 2 ms). 
        When an element of rho is one, this indicates the presence of a spike at the corresponding time, whereas a zero value indicates no spike. 
    The variable stim gives the sequence of stimulus values at the sampled times.'''


'''Exercise. H1 neuron’s firing rate
Load in the MAT-file H1_neuron.mat that holds data from a fly H1 neuron response to an approximate white-noise visual motion stimulus. Path to the file: "exercises/data/H1_neuron.mat";
Make a plot of a stimulus values (line plot) 
Add vertical lines for those time points when the spike occurred. 
Use only first 250 observations for this plot for simplicity. '''

# import the file
h1_data = loadmat("exercises/data/H1_neuron.mat", squeeze_me=True)

# amount of timepoints to keep
n = 250
# slice of an `rho` array
spikes = h1_data["rho"][:n]
# slice of an `stim` array
stimulus = h1_data["stim"][:n]
# time points for the slice
timepoints = np.arange(n)
# select only those when spike occurred
timepoints = timepoints[spikes == 1]

#plt.figure(figsize=(10,5), facecolor="white")
#plt.vlines(
#    x=timepoints,        # time points when spike occurred
#    ymin=stimulus.min(), # lower end of a line
#    ymax=stimulus.max(), # upper end of a line
#    linewidth=1,
#    label="Spike")
# plot stimulus values
#plt.plot(stimulus, label="Stimulus")
#plt.legend()
#plt.title("Fly H1 neuron response to\napproximate white-noise visual motion stimulus")
#plt.xlabel('Time step')
#plt.ylabel('Stimulus value, mV')
#plt.show()

'''Working with JSON files'''

'''JSON (JavaScript Object Notation) is an open standard file format, and data interchange format.
Using human-readable text to store and transmit data objects consisting of attribute–value pairs and array data types (or any other serializable value). 
It is a very common data format, with a diverse range of applications. '''

'''The structure of the file is always in a “key-value” format. 
The values can be strings, lists, or another nested key-value object.

To read/write JSON files in Python we use json package, but the whole procedure is pretty the same as for working with text files. JSON files are loaded as a dictionary into Python.'''

import json

'''Read JSON file'''
#with open(file="path/to/the/file.json", mode="r") as file: # same r as .txt files
#    output = json.load(fp=file)

'''Write JSON file'''
#with open(file="path/to/the/file.json", mode="w") as file: # same w as .txt files
#    output = json.load(fp=file)

'''Exercise
Save a dictionary as a JSON file "dataset_description.json".
Read in the file you just saved.
Save the name of the first author to a variable first_author_name by reaching to it through the output variable (do not just type it in).Keep in mind that in this example value of the Authors key is a list with two dictionaries.'''

import pprint

dataset_description = {
    "Name": "Our cool data set",
    "Authors": [
        {
            "Name": "Bob",
            "Institution": "X"
        },
        {
            "Name": "Ben",
            "Institution": "Y"
        }
    ],
    "Version": "0.0.1"
}

# write JSON file
with open(file="dataset_description.json", mode="w") as file:
    json.dump(obj=dataset_description, fp=file)

# read JSON file
with open(file="dataset_description.json", mode="r") as file: # same r as .txt files
    output = json.load(fp=file)

#pprint.pprint(output)

# get the name of the first author
#first_author_name = output["Authors"][0]["Name"] # you can reach to lower “layers” by specifying the keys of nested dictionaries or indexes of a nested lists, for example <dict name>["key1"]["key2"][<index of a list>]
#print(f"\nFirst author is {first_author_name}")

'''API Requests'''
''' API is Application Programming Interface'''

'''Programmers as clients, create a request, that has 4 main methods:

GET (used most often)
POST
PUT
DELETE

Server sends us data back as a response, that is usually in a JSON or XML formats'''

'''Access: are you allowed to get access?
Request: what do you want to get?
Response: the resulting data'''

'''Open APIs from Space'''
'''There are several package to handle requests in Python, like httplib, urllib2, requests. '''

import requests
import plotly.graph_objects as go
from datetime import datetime

URL = "http://api.open-notify.org/iss-now.json"
response = requests.get(URL)
iss_data = response.json()
fig = go.Figure(
    data=go.Scattergeo(
        lon=[iss_data['iss_position']['longitude']],
        lat=[iss_data['iss_position']['latitude']],
        text=['ISS is over here!'],
        mode='markers',
        marker=dict(color='red')))
fig.update_layout(
    title = f'<b>The Location of ISS @{datetime.fromtimestamp(iss_data["timestamp"])}</b>',
    geo_scope='world')
#fig.show()

#print(response.json()) #  convert raw data into Python dictionary


'''Additional Params'''
#URL = "http://api.open-notify.org/iss-pass.json"

#parameters = {
#    "lat": 52.1205, # Magdeburg, DE
#    "lon": 11.6276,
#    "n": 3}

#response = requests.get(URL, params=parameters)
#print(response.status_code)
#print(response.url)

from tabulate import tabulate

text = '''
Code	Description
200 (OK)	The request was fulfilled.
204 (No response)	Server has received the request but there is no information to send back.
301 (Moved Permanently)	The URL of the requested resource has been changed permanently. The new URL is given in the response.
400 (Bad request)	The request had bad syntax or was inherently impossible to be satisfied.
403 (Forbidden)	The request is for something forbidden. Authorization will not help.
404 (Not found)	The server has not found anything matching the URI given.
500 (Internal Error)	The server encountered an unexpected condition which prevented it from fulfilling the request.
'''

text2 = '''
2xx codes indicate success;
3xx codes indicate redirection;
4xx codes indicate error on the client's side;
5xx codes indicate error on the server's side.
'''
#print(text)

# Parse the text into rows
#lines = text.strip().split('\n')
#headers = lines[0].split('\t')  # Split header by tab
#rows = [line.split('\t') for line in lines[1:]]  # Split each row by tab

# Generate and print the table
#print(tabulate(rows, headers=headers, tablefmt="grid"))

# Parse the text into rows
# print(text2)

'''API requests practice'''

'''Exercise 1. Open APIs From Space
Open Notify is an open-source project to provide a simple programming interface for some of NASA's awesome data. 
Take raw data and turn them into APIs related to space and spacecraft.

There is an API that allows getting the number of people in space at the moment: 
How Many People Are In Space Right Now. There is no authentication for this API, so we don’t need any keys. 
Also, according to the documentation, we cannot specify any additional parameters.

API URL: http://api.open-notify.org/astros.json'''

import requests

# URL of an API request
#URL = "http://api.open-notify.org/astros.json"
# GET method
#response = requests.get(URL)
# check the status code
#print(f"Status code: {response.status_code}")
# convert result to the dictionary
#output = response.json()
# get all the names in a list
#names = []
#for person in output["people"]:
#    names.append(person["name"])

#print(names)

'''Exercise 2. Allen Brain Map
The Allen Institute for Brain Science uses a unique approach to generate data, tools and knowledge for researchers to explore the biological complexity of the mammalian brain. This portal provides access to high quality data and web-based applications created for the benefit of the global research community.

Website

Image download API serves whole and partial two-dimensional images presented on the Allen Brain Atlas Web site. Some images can be downloaded with expression or projection data. Glioblastoma images’ color block and boundary data can also be downloaded.

API URL: http://api.brain-map.org/api/v2/image_download/[SubImage.id]

This API returns an image, not a JSON/XML file and can take several parameters, for example:

quality: integer, the jpeg quality of the returned image. This must be an integer from 0, for the lowest quality, up to as high as 100.
downsample: integer, number of times to downsample the original image. 
    For example, downsample=1 halves the number of pixels of the original image both horizontally and vertically.
width: integer, number of columns in the output image, specified in tier-resolution (desired tier) pixel coordinates.
height: integer, number of rows in the output image, specified in tier-resolution (desired tier) pixel coordinates.
Your task is to:

Get the image for the subject ID 69750516.
Update the parameters of request: set the quality to 50%, set downsample to 2, the resulting image should have the size (5000, 5000)
Print the status code and plot the resulting image.'''

import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#ID = 69750516 # subject ID
# API URL
#URL = f"http://api.brain-map.org/api/v2/image_download/{[ID]}"
# additional parameters for request
#params = {
#    "quality": 50,
#    "downsample": 2,
#    "width": 5000,
#    "height": 5000
#    }

#response = requests.get(URL, params=params)
#print(f"Status code: {response.status_code}")

# convert result binary image to numpy array
#img = Image.open(io.BytesIO(response.content))
#img = np.asarray(img)

# plot the image
#plt.figure(figsize=(7,7), facecolor="white")
#plt.imshow(img)
#plt.axis("off")
#plt.show()

'''Working with Pickled files'''
'https://docs.python.org/3/library/pickle.html' # Docs

'''some big disadvantages of pickled files: 
    The pickle module is NOT secure. Only unpickle data you trust.
    It is Python-only: pickles cannot be loaded in any other programming language (unlike JSON files).'''

'''Reading Pickle files'''
#with open(file="path/to/the/file.pickle", mode="rb") as file: # rb is Read Binary
#    output = pickle.load(file=file)

'''Writing Pickle files'''
#with open(file="path/to/the/file.pickle", mode="wb") as file: # wb is Write Binary
#    pickle.dump(obj=<object to write>, file=file)

'''Exercise
Load in the dataset with fMRI data ("exercises/data/fmri_data.csv"). CSV file is separated by ”;“.
Create a new dictionary frmi with two keys: "parietal" that holds a DataFrame with observations only from the parietal region and "frontal" that holds a DataFrame with observations only from the frontal region.
Save the resulting dictionary to the pickle file "frmi_dict.pickle".
Read in back the resulting pickle file and print the first 5 rows of each of the DataFrames in a dictionary.'''

import pickle
import pandas as pd

# read in the fMRI file
fmri_df = pd.read_csv("exercises/data/fmri_data.csv",sep=';')

# split into two DataFrames according to the region
frmi = {
    "parietal": fmri_df[fmri_df["region"] == "parietal"],
    "frontal": fmri_df[fmri_df["region"] == "frontal"]
}

# write the resulting dictionary as pickle file
with open(file="frmi_dict.pickle", mode="wb") as file:
    pickle.dump(obj=frmi, file=file)

# read in pickle file
with open(file="frmi_dict.pickle", mode="rb") as f:
    output = pickle.load(file=f)

# print out the keys of the loaded dictionary
#print(output['parietal'].head())
#print(output['frontal'].head())

'''Local Files'''

text3 = '''
os.getcwd() - get the current working directory (folder);
os.chdir(path) - change directory;
os.listdir(path=".") - return the content of a directory. If path is not specified, returns the content of current working directory;
os.mkdir(path) - create a new directory();
os.rmdir(path) - remove a directory;
os.remove(path) - remove a file;
os.path.join(dirpath, name) - extend the path to the directory/file. '''

#print(text3)

# Parse the text into rows
lines = text3.strip().split('-')
headers = lines[0].split('\t')  # Split header by tab
rows = [line.split('\t') for line in lines[1:]]  # Split each row by tab

# Generate and print the table
print(tabulate(rows, headers=headers, tablefmt="grid"))


import os

# get the current working directory and save it to a variable
cwd = os.getcwd()
print(f"Initial CWD: {cwd}")

path_to_data = 'exercises/data'
# extend the current path with "path_to_data" part
new_cwd = os.path.join(cwd, path_to_data)
# change the CWD to new_cwd
os.chdir(new_cwd)
print(f"Changing CWD to {new_cwd}")

# get the file names of a CWD
fnames = os.listdir()

# calculate how many CSV files are there in CWD
n_csv = 0
for file_name in fnames:
    if file_name.endswith(".csv"):
        n_csv += 1

#print(f"There are {n_csv} CSV files in {new_cwd} directory.")