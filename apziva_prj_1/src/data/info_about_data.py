import pandas as pd
import numpy as np

import csv
filename="happinesssurvey3.1.csv"

fields = []
rows = []
columns = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)

    for row in csvreader:
        rows.append(row)

    for col in csvreader:
        columns.append(col)
    
    print("Total Number of Rows: %d"%(csvreader.line_num))
print('Field Names Are: '+', '.join(field for fields in fields))
print('\nData:\n')
for row in rows[:126]:
    for col in row:
        print("%10s"%col,end=" "),
    print('\n')