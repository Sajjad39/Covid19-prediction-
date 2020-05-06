#!/usr/bin/env python
# coding: utf-8

# This analysis was done by Ali S. Razavian
# Link- https://medium.com/@ali_razavian/covid-19-from-a-data-scientists-perspective-95bd4e84843b
# 

# In[1]:


import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


url = 'https://covid.ourworldindata.org/data/ecdc/new_deaths.csv'
stream = requests.get(url).content
covid_data = pd.read_csv(io.StringIO(stream.decode('utf-8'))).fillna(0)


# In[12]:


country_population = { 'United States': 331002651, 'Canada': 37894799, 'Italy': 60461826, 'Spain': 46754778, 'France': 65273511,'United Kingdom': 67886011, 'Netherlands': 17134872, 'Germany': 83783942, 'Belgium': 11589623, 'Switzerland': 8654622, 'Turkey': 84339067, 'Sweden': 10099265, 'Portugal': 10196709, 'Austria': 9006398, 'Denmark': 5792202, 'Romania': 19237691,}


records = {}
for name, population in country_population.items():
    daily_death = covid_data[name].rolling(window=3,center=True).mean().values  
    daily_death = daily_death / population
    daily_death = daily_death[daily_death > 1e-8]
    records[name] = daily_death


# In[13]:


def fit(records, alpha_1, alpha_2, left_out_country=None):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    for k, v in records.items():
        if left_out_country is not None and k == left_out_country:
            X = val_x
            Y = val_y
        else:
            X = train_x
            Y = train_y
    for i in range(len(v)):
        X.append([i**(x/5) for x in range(0, 10)])
        Y.append(np.log(v[i]))
    brr = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, normalize=False).fit(np.array(train_x), np.array(train_y))
    if left_out_country is None:
        return np.exp(brr.predict(np.array([[i**(x/5) for x in range(0, 10)] for i in range(70)])))
    else:
        return brr.score(val_x, val_y)
best_alpha_1 = None
best_alpha_2 = None
best_score = 0
for alpha_2 in np.exp(range(-5, 15)):
    for alpha_1 in np.exp(range(-5, 15)):
    score = 0
    for k in records.keys():
        score += fit(records, alpha_1=alpha_1,alpha_2=alpha_2, left_out_country=k)
    if best_score < score:
        best_alpha_1 = alpha_1
        best_alpha_2 = alpha_2
        best_score = score
prediction = fit(records, alpha_1=best_alpha_1,alpha_2=best_alpha_2)


# In[ ]:




