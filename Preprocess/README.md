Instruction
===========================
This is instruction of how to run may code.
****
Author: Xin Jin

Email: jin.967@osu.edu

****
# Files
In the folder 'Preprocess', there are 3 files: dataset.txt, preprocess.py, TxtToCsv.py.

# Running Steps

Step 1. Run: python TxtToCsv.py 

I have collected all data from the 2 datasets and record them into a new file: dataset.tex. You need only run python TxtToCsv.py to convert .txt file to .csv file.

You will get file: dataset.csv

Step 2. Run: python preprocess.py

This is the main file of this project. You will get the result.csv file.

In the result.csv file, the columns are feature vertor and the index is the original sentences.

# Outcomes

1. After Step 1, you will get 'dataset.csv' file (in you current folder). It will be used in step 2.

2. After Step 2, you will get 'result.csv' file ((in you current folder)). It  is the dataFrame format data. Its columns is the feature vector. Its indexes are the original sentences.