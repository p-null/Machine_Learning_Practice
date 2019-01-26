
# coding: utf-8

# # Exploration

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import log_loss
# In[2]:


df = pd.read_csv('data/data_1.csv')



def likelihood_ratio_test(features_alternate, labels, model, features_null=None):
    """
    Parameters:
    ----
    
    
    
    Returns:
    ----
    
    
    """
    labels = np.array(labels)
    features_alternate = np.array(features_alternate)
    
    if features_null is not None:
        features_null = np.array(features_null)
        
        assert(features_null.shape[1] <= features_alternate.shape[1])        
        model.fit(features_null, labels)
        null_prob = model.predict(features_null)
        df = features_alternate.shape[1] - features_null.shape[1]
    else:
        null_prob = sum(labels) / float(labels.shape[0]) * np.ones(labels.shape)
        df = features_alternate.shape[1]
    
    model.fit(features_alternate, labels)
    alt_prob = model.predict(features_alternate)

    alt_log_likelihood = -log_loss(labels,
                                   alt_prob,
                                   normalize=False)
    null_log_likelihood = -log_loss(labels,
                                    null_prob,
                                    normalize=False)

    G = 2 * (alt_log_likelihood - null_log_likelihood)
    p_value = chi2.sf(G, df)

    return p_value


# In[61]:


linear_reg = LinearRegression()

features_alternate =  df[['Abdomen', 'Chest']].values
features_null = df[['Abdomen']].values
labels = df[['%Body Fat']].values.reshape(-1,1)


# In[62]:


p_value = likelihood_ratio_test(features_alternate, labels, linear_reg, features_null)


# We want to test how well '%Body fat' increases with 'Abdomen' (null model) compared to how well '%Body fat' increases with 'Abdomen' and 'Chest' (alternative model).

# In[38]:
