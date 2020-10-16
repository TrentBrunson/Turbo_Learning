#%%
# load libraries
import os
import csv
import pandas as pd
import numpy as np
import glob
import shutil

from sqlalchemy import create_engine
from config import db_password
# %%
# read over data folder and open each csv
# merge into single csv for data cleaning
csv_destdir = ('/data/')
AllData = "data/AllData.csv"
allFiles = glob.glob("**/*.csv", recursive = True)
data = []
# --------------------------------------------------

# allFiles = glob.glob('**/*.csv', recursive= True)
# with open(AllData, 'wb') as outfile:
#     for i, fname in enumerate(allFiles):
#         with open(fname, 'rb') as infile:
#             shutil.copyfileobj(infile, outfile)

# ---------------------------------------


for i, fname in enumerate(allFiles):
    with open(fname, 'r') as infile:
        read_lines = infile.readlines()
        read_lines = ['{0},{1}'.format(line.rstrip('\n'),fname) for line in read_lines]
        data += read_lines

with open(AllData, 'w') as outfile:
    outfile.write('\n'.join(data))
#%%
# drop columns
comlumn_drop_list = []

