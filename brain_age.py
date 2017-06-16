import os
import sys
import numpy as np
import pandas as pd
import csv as csv
from collections import Counter

'''df = pd.read_csv('IXI.csv', header = 0)
df = df.drop(['SEX_ID', 'HEIGHT', 'WEIGHT', 'ETHNIC_ID', 'MARITAL_ID', 'OCCUPATION_ID', 'QUALIFICATION_ID', 'DOB', 'STUDY_DATE'], axis=1)
df = df.drop(df[df.DATE_AVAILABLE < 1].index)
print df.info()
df.to_csv('mriData.csv')

df = pd.read_csv('mriData.csv', header = 0)
df = df.drop(['DATE_AVAILABLE'], axis=1)
df.to_csv('mriData.csv')'''

df = pd.read_csv('mriData.csv', header=0)
#print df.describe()
ids = df['IXI_ID'].values
print len(ids)

fpaths = []
for f in os.listdir('Brain_Image'):
    fpaths.append('Brain_Image/' + f)
print type(fpaths)
print len(fpaths)


a = []
for i in range(len(fpaths)):
	a.append(fpaths[i][15:18])
a = map(int, a)
#print a[:5]
def returnNotMatches(a, b):
    return [[x for x in a if x not in b], [x for x in b if x not in a]]
print returnNotMatches(ids, a)

c = [item for item, count in Counter(ids).iteritems() if count > 1]
print c