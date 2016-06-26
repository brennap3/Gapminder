# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 12:27:51 2016

@author: Peter
"""

import os

import pandas
import numpy
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import sys; print(sys.path)
from seaborn import *
import seaborn as sns
import ggplot
from ggplot import *
import scipy

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
 # Feature Importance
from sklearn.ensemble import ExtraTreesClassifier


import pydot

import graphviz



apath='C:\Users\Peter\Desktop\Gapminder'
print(apath)
os.chdir('C:\Users\Peter\Desktop\Gapminder')
##check the directory has changed
os.getcwd()
##read in the file
data = pandas.read_csv('gapminder.csv', low_memory=False)

##lets convert the data to  numeric
data['incomeperperson'] = data['incomeperperson'].convert_objects(convert_numeric=True)
data['alcconsumption'] = data['alcconsumption'].convert_objects(convert_numeric=True)
data['armedforcesrate'] = data['armedforcesrate'].convert_objects(convert_numeric=True)
data['breastcancerper100th'] = data['breastcancerper100th'].convert_objects(convert_numeric=True)
data['co2emissions'] = data['co2emissions'].convert_objects(convert_numeric=True)

data['femaleemployrate'] = data['femaleemployrate'].convert_objects(convert_numeric=True)
data['hivrate'] = data['hivrate'].convert_objects(convert_numeric=True)
data['internetuserate'] = data['internetuserate'].convert_objects(convert_numeric=True)
data['lifeexpectancy'] = data['lifeexpectancy'].convert_objects(convert_numeric=True)
data['oilperperson'] = data['oilperperson'].convert_objects(convert_numeric=True)

data['polityscore'] = data['polityscore'].convert_objects(convert_numeric=True)
data['relectricperperson'] = data['relectricperperson'].convert_objects(convert_numeric=True)
data['suicideper100th'] = data['suicideper100th'].convert_objects(convert_numeric=True)
data['employrate'] = data['employrate'].convert_objects(convert_numeric=True)
data['urbanrate'] = data['urbanrate'].convert_objects(convert_numeric=True)


bins = [0, 1000, 5000, 10000, 20000,50000,200000]
group_names = ['Very Low Income,0-1000', 'Low Income,1000-5000', 'Okay Income,5000-10000', 'Good Income,10000-20000','Great Income,20000-50000','50,000-200,000']

categories = pandas.cut(data['incomeperperson'], bins, labels=group_names)
data['categories'] = pandas.cut(data['incomeperperson'], bins, labels=group_names)


##data.dtypes chk

##now encode european countries


##

##Ok lets see what the best features are
##note to one self HIV rates missing froma lot of countries


datatransparency = pandas.read_csv('CPI_2015_DATA.csv', low_memory=False)


##w['female'] = w['female'].map({'female': 1, 'male': 0})
datatransparency.columns.values
data.columns.values

## Dont use map
## datatransparency['Country']= datatransparency['Country'].map(
## {"The United States Of America":"United States",
##  "C“te dïIvoire":"Cote d'Ivoire",
##  "Korea (South)":"Korea, Rep.",
##  "Korea (North)":"Korea, Dem. Rep.",
##  "Czech Republic":"Czech Rep.",
##  "Democratic Republic of the Congo":"Congo, Dem. Rep.",
##  "The FYR of Macedonia": "Macedonia, FYR",
##  "Hong Kong":"Hong Kong, China"
## })
 

def country_consistent (row):
     if row['Country'] == "The United States Of America" :
      return "United Sates"
     elif row['Country'] == "C“te dïIvoire" :
      return "Cote d'Ivoire"   
     elif row['Country'] == "Korea (South)" :
      return "Korea, Rep."
     elif row['Country'] == "Korea (North)" :
      return "Korea, Dem. Rep." 
     elif row['Country'] == "Korea (South)" :
      return "Korea, Rep."
     elif row['Country'] == "Czech Republic" :
      return "Czech Rep."
     elif row['Country'] == "Democratic Republic of the Congo" :
      return "Congo, Dem. Rep."
     elif  row['Country'] == "The FYR of Macedonia" :
      return "Macedonia, FYR"
     elif  row['Country'] == "Hong Kong" :
      return "Hong Kong, China"     
     else :
      return row['Country']

datatransparency['Country'] = datatransparency.apply (lambda row: country_consistent(row),axis=1)


##calculate the age of NATO countries
##data['Years_In_Nato'] = data.apply (lambda row: AGE_YEARS (row),axis=1)
##

 ##ok after eyeballing in excel they all look ok

##merge the two datasets

datafullset=data.merge(datatransparency,left_on='country',right_on='Country',how='left')
 
datafullset.columns.values

datafullset.count


##        'country', 'incomeperperson', 'alcconsumption', 'armedforcesrate',
##       'breastcancerper100th', 'co2emissions', 'femaleemployrate',
##       'hivrate', 'internetuserate', 'lifeexpectancy', 'oilperperson',
##       'polityscore', 'relectricperperson', 'suicideper100th',
##       'employrate', 'urbanrate', 'categories', 'European', 'African',
##       'Asian', 'Mid_East', 'North_American', 'Carribean_Central_America',
##       'OPEC', 'Arab_League', 'ASEAN_ARF', 'South_American',
##       'Is_Nato_Country', 'Year_Joined_Nato', 'Eu_Member', 'Years_In_Nato',
##       'NATO_EU_MEMBERSHIP', 'polityscore_cat', 'Rank', 'CPI2015',
##       'Country', 'Region', 'wbcode', 'World Bank CPIA',
##       'World Economic Forum EOS', 'Bertelsmann Foundation TI',
##       'African Dev Bank', 'IMD World Competitiveness Yearbook',
##       'Bertelsmann Foundation SGI', 'World Justice Project ROL',
##       'PRS International Country Risk Guide',
##       'Economist Intelligence Unit', 'IHS Global Insight',
##       'PERC Asia Risk Guide', 'Freedom House NIT', 'CPI2015(2)', 'Rank2',
##       'Number of Sources', 'Std Deviation of Sources', 'Standard Error',
##       'Minimum', 'Maximum', 'Lower CI', 'Upper CI', 'Country2'


## we are going to not include information

data2=datafullset[['country','incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide',
         'polityscore']]

data2.count

data_clean2_pre=data2.dropna()

data_clean2_pre.count

data_clean2 =  data_clean2_pre[['incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide',
         'polityscore']]
         ## drop all na values cant handle nulls

data_clean2.count

from sklearn import preprocessing

## standardize the dataset

data_clean2['incomeperperson']=preprocessing.scale(data_clean2['incomeperperson'].astype('float64'))
data_clean2['alcconsumption']=preprocessing.scale(data_clean2['alcconsumption'].astype('float64'))
data_clean2['armedforcesrate']=preprocessing.scale(data_clean2['armedforcesrate'].astype('float64'))
data_clean2['femaleemployrate']=preprocessing.scale(data_clean2['femaleemployrate'].astype('float64'))
data_clean2['internetuserate']=preprocessing.scale(data_clean2['internetuserate'].astype('float64'))
data_clean2['lifeexpectancy']=preprocessing.scale(data_clean2['lifeexpectancy'].astype('float64'))
data_clean2['employrate']=preprocessing.scale(data_clean2['employrate'].astype('float64'))
data_clean2['CPI2015']=preprocessing.scale(data_clean2['CPI2015'].astype('float64'))
data_clean2['World Economic Forum EOS']=preprocessing.scale(data_clean2['World Economic Forum EOS'].astype('float64'))
data_clean2['PRS International Country Risk Guide']=preprocessing.scale(data_clean2['PRS International Country Risk Guide'].astype('float64'))
data_clean2['polityscore']=preprocessing.scale(data_clean2['polityscore'].astype('float64'))

##

###
###check the standardization worked

###
###

                                                              
from sklearn import preprocessing
from sklearn.cluster import KMeans

##I will not split the dataset

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,20)
meandist=[]

##
#
##


for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(data_clean2)
    clusassign=model.predict(data_clean2)
    meandist.append(sum(np.min(cdist(data_clean2, model.cluster_centers_, 'euclidean'), axis=1)) 
    / data_clean2.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')



####From the scree plot lets see what we can see
#principal component analysis
## lets create a scree plot to see how much of the variance our 

 

###

model2=KMeans(n_clusters=2)
model2.fit(data_clean2)
clusassign2=model2.predict(data_clean2)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(data_clean2)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model2.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show()




# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(data_clean2)
clusassign=model3.predict(data_clean2)
# plot clusters


pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(data_clean2)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

## Interpret 5 cluster solution

model5=KMeans(n_clusters=5)
model5.fit(data_clean2)
clusassign5=model2.predict(data_clean2)
# plot clusters


pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(data_clean2)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model5.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 5 Clusters')
plt.show()


##now try t-sme


X_tsne3 = TSNE(learning_rate=100).fit_transform(data_clean2)
plt.scatter(X_tsne3[:, 0], X_tsne3[:, 1], c=model3.labels_)
plt.title("T-sne Plot categories Polity scores for 3 cluster")


X_tsne5 = TSNE(learning_rate=100).fit_transform(data_clean2)
plt.scatter(X_tsne5[:, 0], X_tsne5[:, 1], c=model5.labels_)
plt.title("T-sne Plot categories Polity scores for 5 cluster")

##these are not effective at finding linear or non linear combinations of variables


# k=3 or k=5 gives the best split 
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
data_clean2.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(data_clean2['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']
newclus['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_clust_names=pd.merge(data_clean2, newclus, on='index')
merged_clust_names.head(n=100)
merged_clust_names.columns.names
# cluster frequencies
###
#
#
###
data_clean2_pre.reset_index(level=0, inplace=True)
countrylist=list(data_clean2_pre['country'])
countrylistindex=list(data_clean2_pre['index'])
newcountrylist=dict(zip(countrylistindex,countrylist))
newcountry=DataFrame.from_dict(newcountrylist,orient='index')
newcountry.columns = ['country']
newcountry['country']
newcountry.reset_index(level=0, inplace=True)
newcountry['country']
###
#
#
###
merged_clust_names_country=pd.merge(merged_clust_names, newcountry, on='index')
merged_clust_names_country[['country']]
##we can quickly see from applying 

clustergrp = merged_clust_names_country.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)


'''
print(clustergrp)

              index  incomeperperson  alcconsumption  armedforcesrate  \
cluster                                                                 
0        108.781250         1.235069        0.615147         0.038096   
1        100.423077        -0.407125       -0.154069         0.255742   
2        108.307692        -0.705835       -0.448967        -0.558371   

         femaleemployrate  internetuserate  lifeexpectancy  employrate  \
cluster                                                                  
0                0.132927         1.260777        0.909442   -0.020729   
1               -0.698979        -0.291682       -0.000986   -0.610994   
2                1.234355        -0.968362       -1.117341    1.247499   

          CPI2015  World Economic Forum EOS  \
cluster                                       
0        1.349325                  1.234625   
1       -0.449433                 -0.382780   
2       -0.761841                 -0.753979   

         PRS International Country Risk Guide  polityscore  
cluster                                                     
0                                    1.311189     0.529000  
1                                   -0.478346    -0.058928  
2                                   -0.657079    -0.533222  
'''

# validate clusters in training data by examining cluster differences in GPA using ANOVA
# first have to merge GPA with clustering variables and cluster assignment data



import patsy
import pandas
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

##sub1.apply(lambda x: pd.to_numeric('cluster', errors='ignore'))

polity_score_train, polity_score_test = train_test_split(merged_clust_names_country, test_size=.3, random_state=123)

sub1 = polity_score_train[['cluster','polityscore']] 
sub1.dtypes


sub1['cluster_str'] = sub1['cluster'].astype(str)


polityscoremod= smf.ols(formula='polityscore ~ C(cluster)', data=sub1).fit()
print (polityscoremod.summary())
'''
print (gpamod.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            polityscore   R-squared:                       0.154
Model:                            OLS   Adj. R-squared:                  0.131
Method:                 Least Squares   F-statistic:                     6.730
Date:                Wed, 22 Jun 2016   Prob (F-statistic):            0.00206
Time:                        22:37:15   Log-Likelihood:                -104.11
No. Observations:                  77   AIC:                             214.2
Df Residuals:                      74   BIC:                             221.2
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
===================================================================================
                      coef    std err          t      P>|t|      [95.0% Conf. Int.]
-----------------------------------------------------------------------------------
Intercept           0.5299      0.199      2.664      0.009         0.134     0.926
C(cluster)[T.1]    -0.5851      0.256     -2.285      0.025        -1.095    -0.075
C(cluster)[T.2]    -1.0770      0.296     -3.641      0.001        -1.666    -0.488
==============================================================================
Omnibus:                       24.509   Durbin-Watson:                   1.793
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               33.820
Skew:                          -1.444   Prob(JB):                     4.53e-08
Kurtosis:                       4.483   Cond. No.                         4.02
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
'''
print ('means for PolityScore by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for polityscore by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['polityscore'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())

####
#######
#######
####

#########
###
#
''''
Multiple Comparison of Means - Tukey HSD,FWER=0.05
=============================================
group1 group2 meandiff  lower   upper  reject
---------------------------------------------
  0      1    -0.5851  -1.1976  0.0274 False 
  0      2     -1.077  -1.7845 -0.3696  True 
  1      2    -0.4919  -1.1422  0.1583 False 
---------------------------------------------

Tukey HSD

'''

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

### lets visualize this

sub1pivot=sub1.copy
sub2=pd.DataFrame(sub1[['cluster_str','polityscore']])
sub2['idx'] = sub2.groupby('cluster_str').cumcount()

sub2.reset_index()
sub2.dtypes
pivoted = sub2.pivot(columns='cluster_str', values='polityscore')
pivoted.columns.names
pivoted.reset_index()
pivoted[['0','1','2']]
pivoted.dtypes
pivotedplt=pivoted[['0','1','2']].reset_index()

ggplot(sub1, aes(x='cluster', y='polityscore')) + geom_boxplot() +ggtitle("boxplot of Polity scores -versus-cluster (3 cluster model)")

###
#
###
'''
Okay there seems to be a problem with this we are only seeing a statistically significant at the p = 0.05 level
We can even see from the boxplot there is an overlap of the plotly scores. 
'''

##From the scree plot lets see what we can see


pca = PCA(n_components=11)
 
pca.fit(data_clean2[['incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide',
         'polityscore']])

#The amount of variance that each PC explains
 varianceexplainedbyPCACOMP= pca.explained_variance_ratio_
##
 PCACUMPLOT=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


plt.plot(PCACUMPLOT)
plt.title("Cumulative varaince of components against number of principal components")
plt.xlabel("Princiapl Component")
plt.ylabel("Cumulative varaince explain")

## wow the first two components hold more than 80% of the variance

##

tran_pca = pca.fit(data_clean2[['incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide',
         'polityscore']]).transform(data_clean2[['incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide',
         'polityscore']])
 
df_pca = pd.DataFrame(tran_pca)

df_pca.columns = [['pc1', 'pc2', 'pc3', 'pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']]
df_pca['y'] = merged_clust_names_country[['cluster']]
df_pca.head()

##lets create a biplot

import seaborn as sns

np_cluster=merged_clust_names_country[['cluster']].as_matrix()

# Scatter plot based and assigne color based on 'label - y'
sns.lmplot('pc1', 'pc2', data=df_pca, fit_reg = False,  size = 15, hue='y', scatter_kws={"s": 100})


# set the maximum variance of the first two PCs
# this will be the end point of the arrow of each **original features**
xvector = pca.components_[0]
yvector = pca.components_[1]
 
X=data_clean2[['incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide',
         'polityscore']]
# value of the first two PCs, set the x, y axis boundary
xs = pca.transform(X)[:,0]
ys = pca.transform(X)[:,1]

for i in range(len(xvector)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
    # we can adjust length and the size of the arrow
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.005, head_width=0.05)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(X.columns.values)[i], color='r')

np_df = data_clean2_pre[['country']].as_matrix()
##np_df[0] rember numpy arrays are 0 indexed
for i in range(len(xs)):
    plt.text(xs[i]*1.08, ys[i]*1.08, np_df[i],color='b') # index number of each observations
plt.title('PCA Plot of first PCs')



########
###
##lets try 5 clusters
#
#
##
#######
## Interpret 5 cluster solution

model5=KMeans(n_clusters=5)
model5.fit(data_clean2)
clusassign5=model2.predict(data_clean2)


# create a list that has the new index variable
cluslist5=list(data_clean2['index'])
# create a list of cluster assignments
labels5=list(model5.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist5=dict(zip(cluslist5, labels5))
newlist5
# convert newlist dictionary to a dataframe
newclus5=DataFrame.from_dict(newlist5, orient='index')
newclus5
# rename the cluster assignment column
newclus5.columns = ['cluster']
newclus5['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus5.reset_index(level=0, inplace=True)

newclus5.columns.values

# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_clust_names5=pd.merge(data_clean2, newclus5,on='index')
merged_clust_names5.head(n=100)
merged_clust_names5.columns.names
# cluster frequencies
###
#
#
###
##data_clean2_pre.reset_index(level=0, inplace=True)
countrylist=list(data_clean2_pre['country'])
countrylistindex=list(data_clean2_pre['index'])
newcountrylist=dict(zip(countrylistindex,countrylist))
newcountry=DataFrame.from_dict(newcountrylist,orient='index')
newcountry.columns = ['country']
newcountry['country']
newcountry.reset_index(level=0, inplace=True)
newcountry['country']
###
#
#
###
merged_clust_names_country5=pd.merge(merged_clust_names5, newcountry, on='index')
merged_clust_names_country5[['country']]
##we can quickly see from applying 

clustergrp5 = merged_clust_names_country5.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp5)
'''
Clustering variable means by cluster
              index  incomeperperson  alcconsumption  armedforcesrate  \
cluster                                                                 
0        106.210526         1.925011        0.360945        -0.191434   
1        111.153846        -0.304801       -1.218275         1.514747   
2        108.291667        -0.727085       -0.492013        -0.600894   
3        102.484848        -0.503862       -0.074974        -0.272969   
4         98.809524         0.069748        1.107717         0.351189   

         femaleemployrate  internetuserate  lifeexpectancy  employrate  \
cluster                                                                  
0                0.255068         1.530205        1.029620    0.226588   
1               -1.745974        -0.216228        0.192075   -1.116786   
2                1.287686        -0.991900       -1.184508    1.312131   
3               -0.244027        -0.502937       -0.209858   -0.197278   
4               -0.238104         0.673314        0.633036   -0.703234   

          CPI2015  World Economic Forum EOS  \
cluster                                       
0        1.736917                  1.701811   
1       -0.439791                 -0.032307   
2       -0.777429                 -0.777062   
3       -0.532239                 -0.524499   
4        0.425622                  0.192549   

         PRS International Country Risk Guide  polityscore  
cluster                                                     
0                                    1.718800     0.372376  
1                                   -0.447996    -1.310539  
2                                   -0.655673    -0.591411  
3                                   -0.558362     0.279227  
4                                    0.348993     0.711488  
'''

######
###
#
###
#####

##sub1.apply(lambda x: pd.to_numeric('cluster', errors='ignore'))

polity_score_train_cl5, polity_score_test_cl5 = train_test_split(merged_clust_names_country5, test_size=.3, random_state=123)

sub2 = polity_score_train_cl5[['cluster','polityscore']] 
sub2.dtypes

ggplot(sub2, aes(x='cluster', y='polityscore')) + geom_boxplot() +ggtitle("boxplot of Polity scores -versus-cluster (5 cluster model)")


sub2['cluster_str'] = sub2['cluster'].astype(str)


polityscoremod= smf.ols(formula='polityscore ~ C(cluster_str)', data=sub2).fit()
print (polityscoremod.summary())

''''
print (polityscoremod.summary())

                            OLS Regression Results                            
==============================================================================
Dep. Variable:            polityscore   R-squared:                       0.384
Model:                            OLS   Adj. R-squared:                  0.350
Method:                 Least Squares   F-statistic:                     11.22
Date:                Sat, 25 Jun 2016   Prob (F-statistic):           3.95e-07
Time:                        02:41:52   Log-Likelihood:                -91.891
No. Observations:                  77   AIC:                             193.8
Df Residuals:                      72   BIC:                             205.5
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
=======================================================================================
                          coef    std err          t      P>|t|      [95.0% Conf. Int.]
---------------------------------------------------------------------------------------
Intercept               0.3820      0.213      1.793      0.077        -0.043     0.807
C(cluster_str)[T.1]    -1.4672      0.337     -4.355      0.000        -2.139    -0.796
C(cluster_str)[T.2]    -0.9972      0.289     -3.456      0.001        -1.572    -0.422
C(cluster_str)[T.3]    -0.1227      0.282     -0.435      0.665        -0.685     0.439
C(cluster_str)[T.4]     0.3947      0.307      1.287      0.202        -0.217     1.006
==============================================================================
Omnibus:                       29.390   Durbin-Watson:                   1.957
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               54.126
Skew:                          -1.421   Prob(JB):                     1.76e-12
Kurtosis:                       5.965   Cond. No.                         5.97
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
''''

##yep yep we can reject the NUll hypothesis at 0.05 level

mc5 = multi.MultiComparison(sub2['polityscore'], sub2['cluster_str'])
res5 = mc5.tukeyhsd()
print(res5.summary())

'''''
mc5 = multi.MultiComparison(sub2['polityscore'], sub2['cluster_str'])
res5 = mc5.tukeyhsd()
print(res5.summary())

Multiple Comparison of Means - Tukey HSD,FWER=0.05
=============================================
group1 group2 meandiff  lower   upper  reject
---------------------------------------------
  0      1    -1.4672   -2.41  -0.5245  True 
  0      2    -0.9972  -1.8045 -0.1898  True 
  0      3    -0.1227  -0.9115  0.666  False 
  0      4     0.3947  -0.4634  1.2529 False 
  1      2      0.47   -0.4408  1.3809 False 
  1      3     1.3445   0.4501  2.2389  True 
  1      4     1.862    0.9058  2.8181  True 
  2      3     0.8744   0.1242  1.6247  True 
  2      4     1.3919   0.569   2.2149  True 
  3      4     0.5175  -0.2872  1.3222 False 
---------------------------------------------

'''''



########
####
##
####
#######



df_pca['y'] = merged_clust_names_country5[['cluster']]
df_pca.head(109)

##lets create a biplot


# Scatter plot based and assigne color based on 'label - y'
sns.lmplot('pc1', 'pc2', data=df_pca, fit_reg = False,  size = 15, hue='y', scatter_kws={"s": 100})


# set the maximum variance of the first two PCs
# this will be the end point of the arrow of each **original features**
xvector = pca.components_[0]
yvector = pca.components_[1]
 
X=data_clean2[['incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide',
         'polityscore']]
# value of the first two PCs, set the x, y axis boundary
xs = pca.transform(X)[:,0]
ys = pca.transform(X)[:,1]

for i in range(len(xvector)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
    # we can adjust length and the size of the arrow
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.005, head_width=0.05)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(X.columns.values)[i], color='r')

np_df = data_clean2_pre[['country']].as_matrix()
##np_df[0] rember numpy arrays are 0 indexed
for i in range(len(xs)):
    plt.text(xs[i]*1.08, ys[i]*1.08, np_df[i],color='b') # index number of each observations
plt.title('PCA Plot of first PCs PCA1 and PCA2')



#####
##
#
##
#####

'''
Besides looking at Just the PCA plots lets look at another dimension reduction technique t-sne and visualize our data that way

'''

from sklearn.manifold import TSNE

data2tsne=datafullset[['country','incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide',
         'polityscore']]
         
##run categorization on polityscore
         
         
def polityscore_cat (row):
   if (row['polityscore'] >=6 and row['polityscore'] <= 10 ) :
      return 1 ##democracy
   elif (row['polityscore'] >=-5 and row['polityscore'] <= 5 )  :
      return 2 ##anocracy
   elif (row['polityscore'] >=-10 and row['polityscore'] <= -6 )  :
      return 3   ##autocracy
   else :
      return 0 ##unknown


##calculate the age of NATO countries
##data['Years_In_Nato'] = data.apply (lambda row: AGE_YEARS (row),axis=1)
data2tsne['polityscore_cat'] = data2tsne.apply (lambda row: polityscore_cat (row),axis=1)
         
##drop NA values

data2tsne=data2tsne.dropna()
       
##Explore 

polityscoredata=data2tsne[['incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide',
         ]]

polityscoretarget= data2tsne.polityscore_cat


polityscoredata['incomeperperson']=preprocessing.scale(polityscoredata['incomeperperson'].astype('float64'))
polityscoredata['alcconsumption']=preprocessing.scale(polityscoredata['alcconsumption'].astype('float64'))
polityscoredata['armedforcesrate']=preprocessing.scale(polityscoredata['armedforcesrate'].astype('float64'))
polityscoredata['femaleemployrate']=preprocessing.scale(polityscoredata['femaleemployrate'].astype('float64'))
polityscoredata['internetuserate']=preprocessing.scale(polityscoredata['internetuserate'].astype('float64'))
polityscoredata['lifeexpectancy']=preprocessing.scale(polityscoredata['lifeexpectancy'].astype('float64'))
polityscoredata['employrate']=preprocessing.scale(polityscoredata['employrate'].astype('float64'))
polityscoredata['CPI2015']=preprocessing.scale(polityscoredata['CPI2015'].astype('float64'))
polityscoredata['World Economic Forum EOS']=preprocessing.scale(polityscoredata['World Economic Forum EOS'].astype('float64'))
polityscoredata['PRS International Country Risk Guide']=preprocessing.scale(polityscoredata['PRS International Country Risk Guide'].astype('float64'))


X_tsne = TSNE(learning_rate=100).fit_transform(polityscoredata)
X_pca = PCA().fit_transform(polityscoredata)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=polityscoretarget)
plt.title("T-sne Plot categories Polity scores")

## -SNE can help us to decide whether classes are separable in some linear or nonlinear representation. Here we can see that the 3 classes of the Iris dataset can be separated quite easily

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=polityscoretarget)
plt.title("PCA 1 and 2")






