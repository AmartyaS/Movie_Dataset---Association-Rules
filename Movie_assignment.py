# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 06:19:50 2021

@author: Amartya
"""


conda install -c conda -forge mlxtend
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

dataset=pd.read_csv("F:\Softwares\Data Science Assignments\Python-Assignment\Association Rules\\my_movies.csv")
data=dataset.iloc[:,5:15]

#Formation of Association Rules
fritem=apriori(data,min_support=0.2,use_colnames=True)
rules=association_rules(fritem,metric="lift",min_threshold=0.8)

#Eliminating duplicate rules
def tolist(i):
    return (sorted(list(i)))

maxrul= rules.antecedents.apply(tolist) +rules.consequents.apply(tolist)
maxrul=maxrul.apply(sorted)

rul=list(maxrul)

uniquer=[list(m) for m in set(tuple(i) for i in rul)]

indexr=[]
for i in uniquer:
    indexr.append(rul.index(i))

unique_rules=rules.iloc[indexr,:]

unique_rules.sort_values('lift',ascending = False)
