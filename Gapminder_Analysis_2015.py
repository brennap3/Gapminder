# -*- coding: utf-8 -*-
"""
Spyder Editor

This is my Gapminder script file.
it reads in data 
changes the type to numerics
and does some univariate analaysis using
analytic visuals and frequency tables 
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
from ggplot import *


##give the path of our folder
##set the path to wherever you downloaded the dataset
apath='C:\Users\Peter\Desktop\Gapminder'
print(apath)
os.chdir('C:\Users\Peter\Desktop\Gapminder')
##check the directory has changed
os.getcwd()
##read in the file
data = pandas.read_csv('gapminder.csv', low_memory=False)
##check the first 10 columns
##equivalent of r head
data.head(10)
##
columnnames_list_=list(data.columns.values)
## print the list
print(columnnames_list_)

data.dtypes
""" 
country                 object
incomeperperson         object
alcconsumption          object
armedforcesrate         object
breastcancerper100th    object
co2emissions            object
femaleemployrate        object
hivrate                 object
internetuserate         object
lifeexpectancy          object
oilperperson            object
polityscore             object
relectricperperson      object
suicideper100th         object
employrate              object
urbanrate               object
dtype: object
"""
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
## or use the describe function
##this gives us some summary inforation about our data 
##this will give summary info for each row
##this is handy for looking at descriptive univariate analysis

pandas.DataFrame.describe(data)
##subsetting the data
##lets take a subset of data for northern european countries
euro_list=('Belgium','France','Netherlands','Ireland','United Kingdom','Germany','Denmark','Sweden','Norway','Finland')
##subset by getting the index of the countries in above list
##and then getting the data at these indexes
subset_northeastern_europe=data[data['country'].isin(euro_list)]
##lets check the subset operation
subset_northeastern_europe.head(10)
pandas.DataFrame.describe(subset_northeastern_europe)
##as you can see the dataa is very different for North Western Europe


## income per person
print("frequency table of incomeperperson")
p1=data['incomeperperson'].value_counts(sort=False, normalize=True,dropna=False)
print(p1)
##this does not tell me much lets plot the hsitogram
##lets create some categorical varaibles
data['incomeperperson']

bins = [0, 1000, 5000, 10000, 20000,50000,200000]
group_names = ['Very Low Income,0-1000', 'Low Income,1000-5000', 'Okay Income,5000-10000', 'Good Income,10000-20000','Great Income,20000-50000','50,000-200,000']

categories = pandas.cut(data['incomeperperson'], bins, labels=group_names)
data['categories'] = pandas.cut(data['incomeperperson'], bins, labels=group_names)
#print('new categorical variables based on the income per person')
pandas.value_counts(data['categories'])
##from our ouptu we can see that the vast najority of countries
##surveyed are in the low  income or very low income category
'''
Low Income,1000-5000        61
Very Low Income,0-1000      54
Okay Income,5000-10000      28
Great Income,20000-50000    26
Good Income,10000-20000     17
50,000-200,000               4
dtype: int64
'''

##lets rerun the analysis for the northeren europe subset
##subset the data
categories_ne = pandas.cut(subset_northeastern_europe['incomeperperson'], bins, labels=group_names)
subset_northeastern_europe['categories'] = pandas.cut(subset_northeastern_europe['incomeperperson'], bins, labels=group_names)

pandas.value_counts(subset_northeastern_europe['categories'])

##all the north eastern european countries fall into the bracket
##Great Income,20000-50000    10
##its a very homogonous group
'''
Out[63]: 
Great Income,20000-50000    10
50,000-200,000               0
Good Income,10000-20000      0
Okay Income,5000-10000       0
Low Income,1000-5000         0
Very Low Income,0-1000       0
dtype: int64
'''
##now lets do the same for: 
#armedforcesrate         object
#polityscore             object

print("frequency table of armedforcesrate")
p2=data['armedforcesrate'].value_counts(sort=False, normalize=True,dropna=False)
print(p2)
##23% of data is NaN
##only really usefull statistic
armedforcesrate=data['armedforcesrate'][(data['armedforcesrate'] >= 0)].values
bins=100
plt.hist(armedforcesrate, bins, normed=True, color="#F08080", alpha=.5);
##the armed forces rates density distribution is shown
##heavy tailed distribution
##alternatively as a boxplot

plt.boxplot(armedforcesrate)
#alternatively as a violin plot
plt.violinplot(armedforcesrate)


##look at the politly scores
print("frequency table of politly scores")
p3=data['polityscore'].value_counts(sort=True, dropna=False)
print(p3)
##
'''
frequency table of politly scores
NaN    0.244131
 0     0.028169
 1     0.014085
 2     0.014085
 3     0.009390
 4     0.018779
 5     0.032864
 6     0.046948
 7     0.061033
 8     0.089202
 9     0.070423
 10    0.154930
-1     0.018779
-10    0.009390
-9     0.018779
-8     0.009390
-7     0.056338
-6     0.014085
-5     0.009390
-4     0.028169
-3     0.028169
-2     0.023474
'''

##again this does not tell us much
##lets create the 

armedforcesrate=data['armedforcesrate'][(data['armedforcesrate'] >= 0)].values
bins=100
plt.hist(armedforcesrate, bins, normed=True, color="#F08080", alpha=.5);



##Peter Brennan
##13/11/2015
##adding extra categories based upon country
##we will do this by creating functions
'''
five different categories are created   
two of these are created using functions
1 by merging an additional dataframe
and two by comparing to lists

'''
##european countries
'''
##european countries
Albania
Andorra
Armenia
Austria
Azerbaijan
Belarus
Belgium
Bosnia
Bulgaria
Croatia
Cyprus
Czech Republic
Denmark
Estonia
Finland
France
Georgia
Germany
Greece
Hungary
Iceland
Ireland
Italy
Kazakhstan
Kosovo
Latvia
Liechtenstein
Lithuania
Luxembourg
Macedonia
Malta
Moldova
Monaco
Montenegro
Netherlands
Norway
Poland
Portugal
Romania
Russia
San Marino
Serbia
Slovakia
Slovenia
Spain
Sweden
Switzerland
Turkey
Ukraine
United Kingdom
Vatican City (Holy See) leave this out
https://en.wikipedia.org/wiki/List_of_sovereign_states_and_dependent_territories_in_Europe
#####
'''
def EUROPEAN (row):
   if row['country'] == 'Albania' :
      return 'Europe'
   elif row['country'] == 'Andorra' :
      return 'Europe'
   elif row['country'] == 'Armenia' :
      return 'Europe'
   elif row['country'] == 'Azerbaijan' :
      return 'Europe'   
   elif row['country'] == 'Austria' :
      return 'Europe'      
   elif row['country'] == 'Belarus' :
      return 'Europe'
   elif row['country'] == 'Belgium' :
      return 'Europe'   
   elif row['country'] == 'Bosnia' :
      return 'Europe'
   elif row['country'] == 'Bulgaria' :
      return 'Europe'   
   elif row['country'] == 'Croatia' :
      return 'Europe'   
   elif row['country'] == 'Cyprus' :
      return 'Europe'
   elif row['country'] == 'Czech Republic' :
      return 'Europe'
   elif row['country'] == 'Denmark' :
      return 'Europe'  
   elif row['country'] == 'Estonia' :
      return 'Europe'   
   elif row['country'] == 'Finland' :
      return 'Europe'      
   elif row['country'] == 'France' :
      return 'Europe'      
   elif row['country'] == 'Georgia' :
      return 'Europe'            
   elif row['country'] == 'Germany' :
      return 'Europe'            
   elif row['country'] == 'Greece' :
      return 'Europe'      
   elif row['country'] == 'Hungary' :
      return 'Europe'  
   elif row['country'] == 'Iceland' :
      return 'Europe'            
   elif row['country'] == 'Ireland' :
      return 'Europe'                  
   elif row['country'] == 'Italy' :
      return 'Europe'        
   elif row['country'] == 'Kazakhstan' :
      return 'Europe'      
   elif row['country'] == 'Kosovo' :
      return 'Europe'   
   elif row['country'] == 'Latvia' :
       return 'Europe'   
   elif row['country'] == 'Liechtenstein' :
       return 'Europe'   
   elif row['country'] == 'Lithuania' :
       return 'Europe'   
   elif row['country'] == 'Luxembourg' :
       return 'Europe'   
   elif row['country'] == 'Macedonia' :
       return 'Europe'   
   elif row['country'] == 'Malta' :
       return 'Europe'   
   elif row['country'] == 'Moldova' :
       return 'Europe'   
   elif row['country'] == 'Monaco' :
       return 'Europe'   
   elif row['country'] == 'Montenegro' :
       return 'Europe'   
   elif row['country'] == 'Netherlands' :
       return 'Europe'   
   elif row['country'] == 'Norway' :
       return 'Europe'   
   elif row['country'] == 'Poland' :
       return 'Europe'   
   elif row['country'] == 'Portugal' :
       return 'Europe'   
   elif row['country'] == 'Romania' :
       return 'Europe'   
   elif row['country'] == 'Russia' :
       return 'Europe'   
   elif row['country'] == 'San Marino' :
       return 'Europe'   
   elif row['country'] == 'Serbia' :
       return 'Europe'   
   elif row['country'] == 'Slovak REpublic' :    
       return 'Europe'   
   elif row['country'] == 'Slovenia' :    
       return 'Europe'   
   elif row['country'] == 'Spain' :    
       return 'Europe'   
   elif row['country'] == 'Sweden' :
       return 'Europe'   
   elif row['country'] == 'Switzerland' :
       return 'Europe'   
   elif row['country'] == 'Turkey' :       
       return 'Europe'   
   elif row['country'] == 'Ukraine' :
       return 'Europe'   
   elif row['country'] == 'United Kingdom' :
       return 'Europe'          
   else :
      return 'Not-In-Europe'


data['European'] = data.apply (lambda row: EUROPEAN (row),axis=1)

##check it worked

'''
Out[24]: 
Europe            45
Not-In-Europe    168
dtype: int64
'''

data['European'].value_counts(sort=False, dropna=False)


'''
##EU countries list as of 2015
Austria,
Belgium, 
Bulgaria,
Croatia, 
Cyprus, 
Czech Republic, 
Denmark, 
Estonia, 
Finland, 
France, 
Germany, 
Greece, 
Hungary, 
Ireland, 
Italy, 
Latvia, 
Lithuania, 
Luxembourg, 
Malta, 
Netherlands, 
Poland, 
Portugal, 
Romania, 
Slovak Republic, 
Slovenia, 
Spain, 
Sweden,
United Kingdom
## source https://www.gov.uk/eu-eea
##
'''

def EUMEMBER (row):
   if row['country'] == 'Austria' :
      return 'EU'
   elif row['country'] == 'Belgium' :
      return 'EU'
   elif row['country'] == 'Bulgaria' :
      return 'EU'   
   elif row['country'] == 'Croatia' :
      return 'EU'   
   elif row['country'] == 'Cyprus' :
      return 'EU'
   elif row['country'] == 'Czech Republic' :
      return 'EU'
   elif row['country'] == 'Denmark' :
      return 'EU'  
   elif row['country'] == 'Estonia' :
      return 'EU'   
   elif row['country'] == 'Finland' :
      return 'EU'      
   elif row['country'] == 'France' :
      return 'EU'      
   elif row['country'] == 'Germany' :
      return 'EU'            
   elif row['country'] == 'Greece' :
      return 'EU'      
   elif row['country'] == 'Hungary' :
      return 'EU'  
   elif row['country'] == 'Ireland' :
      return 'EU'                  
   elif row['country'] == 'Italy' :
      return 'EU'        
   elif row['country'] == 'Latvia' :
       return 'EU'   
   elif row['country'] == 'Lithuania' :
       return 'EU'   
   elif row['country'] == 'Luxembourg' :
       return 'EU'   
   elif row['country'] == 'Malta' :
       return 'EU'   
   elif row['country'] == 'Netherlands' :
       return 'EU'   
   elif row['country'] == 'Poland' :
       return 'EU'   
   elif row['country'] == 'Portugal' :
       return 'EU'   
   elif row['country'] == 'Romania' :
       return 'EU'   
   elif row['country'] == 'Slovak Republic' :    
       return 'EU'   
   elif row['country'] == 'Slovenia' :    
       return 'EU'   
   elif row['country'] == 'Spain' :    
       return 'EU'   
   elif row['country'] == 'Sweden' :
       return 'EU'   
   elif row['country'] == 'United Kingdom' :
       return 'EU'          
   else :
      return 'Not-In-EU'


data['Eu_Member'] = data.apply (lambda row: EUMEMBER (row),axis=1)
##

###NATO mebers and since when they joined
'''
Albania
2009

Belgium
1949

Bulgaria
2004

Canada
1949

Croatia
2009

Czech Republic
1999

Denmark
1949

Estonia
2004

France
1949

Germany
1955

Greece
1952

Hungary
1999

Iceland
1949

Italy
1949

Latvia
2004

Lithuania
2004

Luxembourg
1949

Netherlands
1949

Norway
1949

Poland
1999

Portugal
1949

Romania
2004

Slovak Republic
2004

Slovenia
2004

Spain
1982

Turkey
1952

United Kingdom
1949

United States
1949
###http://www.nato.int/cps/en/natohq/topics_52044.htm
''''

##NATO data

Nato_Countries = pandas.DataFrame({ 'country' : ('Albania','Belgium','Bulgaria','Canada','Croatia','Czech Republic','Denmark','Estonia','France','Germany','Greece','Hungary','Iceland','Italy','Latvia','Lithuania','Luxembourg','Netherlands','Norway','Poland','Portugal','Romania','Slovak Republic','Slovenia','Spain','Turkey','United Kingdom','United States'),
                     'Year_Joined' : (2009,1949,2004,1949,2009,1999,1949,2004,1949,1955,1952,1999,1949,1949,2004,2004,1949,1949,1949,1999,1949,2004,2004,2004,1982,1952,1949,1949),
                     'Is_Nato_Country' : 'Nato_Member'
                        })

##Enhanced data join NATO data

data=pandas.merge(data, Nato_Countries,how='left',on='country')

##data.columns.values
##check that all column values have been added

data.columns.values
##year joined needs to be renamed
data.rename(columns={'Year_Joined': 'Year_Joined_Nato'}, inplace=True)
## change columns names
data.columns.values
##changes are in place



##calculate the age of countries in NATO
## if not in nato set a decode to set age to -1 

import time
##check how to calc time
print (time.strftime("%Y"))
##write unction to calculate the  age of NATO countries based on the current date
def AGE_YEARS (row):
   current_year=time.strftime("%Y")
   if row['Year_Joined_Nato'] >0 :
      return (int(current_year)-int(row['Year_Joined_Nato']))
   else :
      return -1

##calculate the age of NATO countries
data['Years_In_Nato'] = data.apply (lambda row: AGE_YEARS (row),axis=1)
##calculate the age of NAto countries
print("distribution of years in NATO for Nato Countries")
pp2=data['Years_In_Nato'].value_counts(sort=False, dropna=False)
print(pp2)
##Mean number of years in NATO by european or Non EU
print("Mean years of countries in NATO for European and Non European Countries")
pp3=data[data['Years_In_Nato']>0]['Years_In_Nato'].groupby(data['Eu_Member']).mean()
print(pp3)
##Count of countries in EU who are not in not in NATO
print("Count of countries in NATO for European and Non European Countries")
pp4=data[data['Years_In_Nato']>0]['Years_In_Nato'].groupby(data['Eu_Member']).count()
print(pp4)


##EU 'Eu_Member' Is_Nato_Country 'Nato_Member'


##function tocalculate whether a country is in both the EU and NATO
def EU_NATO (Nato_Membership,EU_Membership):
   if Nato_Membership == 'Nato_Member' and EU_Membership== 'EU' :
      return 'Nato_And_EU'
   elif Nato_Membership == 'Nato_Member' and EU_Membership == 'Not-In-EU':
      return 'Nato_Not_In_EU'
   elif Nato_Membership != 'Nato_Member' and EU_Membership== 'EU' :
      return 'Not_In_Nato_In_EU'    
   else :
      return 'Not_In_Nato_Not_In_EU'
      
      
      
      
      
EU_NATO('Nato_Member','EU')
##test
##apply the function
data['NATO_EU_MEMBERSHIP'] = data.apply (lambda row: EU_NATO(row['Is_Nato_Country'],row['Eu_Member']),axis=1)

print("Nato and EU membership contingency tables")
pp5=data['NATO_EU_MEMBERSHIP'].value_counts(sort=False, dropna=False)
print(pp5)      
##politlyscores for EU countries
print("distribution of politly score for EU countries")
pp6=data[data['Eu_Member']=='EU']['polityscore'].value_counts(sort=True, dropna=False)
print(pp6)

data.dtypes
##check the object type
##cast to a string
data['NATO_EU_MEMBERSHIP']=data['NATO_EU_MEMBERSHIP'].astype(str) 

##ignore commented out code below
'''
ignore only plotting 3 of 4 categoies
ggplot(data, aes(x='armedforcesrate', y='incomeperperson', color="NATO_EU_MEMBERSHIP")) +\
    geom_point() +\
    scale_color_brewer(type='diverging', palette=4) +\
    xlab("armed forces rate") + ylab("IncomePerPerson") + ggtitle("Gapminder")

####
'''

p=ggplot(data, aes(x='armedforcesrate', y='incomeperperson'))
p + geom_point(alpha=0.25) + \
    facet_grid("NATO_EU_MEMBERSHIP")
    





##    subset for europe
data.columns.values
subset_data_europe=data[data['European']=='Europe']

print("Nato and EU membership contingency tables for europe only")
epp5=subset_data_europe['NATO_EU_MEMBERSHIP'].value_counts(sort=False, dropna=False)
print(epp5)      
##europe only
p=ggplot(subset_data_europe, aes(x='armedforcesrate', y='incomeperperson'))
p + geom_point(alpha=0.25) + \
    facet_grid("NATO_EU_MEMBERSHIP")


##
##

import bokeh
##
##

#########
####
##creating-graphs-for-your-data
##Peter Brennan
##20/11/2015
####
########

####
##Income Per Person
####
##

incomeperperson=data['incomeperperson'][(data['incomeperperson'] >= 0)].values
sns.distplot(incomeperperson)
plt.xlabel('incomeperperson')
plt.title('Distribution of Income Per Person Distribution for the world')

data['incomeperperson'].describe()
'''
count       190.000000
mean       8740.966076
std       14262.809083
min         103.775857
25%         748.245151
50%        2553.496056
75%        9379.891165
max      105147.437697
'''

##armedforcesrate

armedforcesrate=data['armedforcesrate'][(data['armedforcesrate'] >= 0)].values
sns.distplot(armedforcesrate)
plt.xlabel('armedforcesrate')
plt.title('Distribution of Armed forces Rate Distribution for the world')


data['armedforcesrate'].describe()
'''
count    164.000000
mean       1.444016
std        1.709008
min        0.000000
25%        0.480907
50%        0.930638
75%        1.611383
max       10.638521
'''
data['armedforcesrate'].describe()

##polityscore
##polity score
##ranges from-10 to 10
##10 being most democratic -10 being 
##total non democratic authoritarian regime



##polityscore

polityscore=data['polityscore'][(data['polityscore'] >= -100)].values
sns.distplot(polityscore)
plt.xlabel('polityscore')
plt.title('Distribution of polityscore (Political freedom) Distribution for the world')

data['polityscore'].describe()
data['polityscore'].median()
data['polityscore'].mode()

'''
polity scores eplained
https://en.wikipedia.org/wiki/Polity_data_series
minimum value	maximum value
autocracies	-10	-6
anocracies	-5	5
democracies	6	10
"Anocracy" is a term used to describe a regime type that is characterized by inherent qualities of political instability and ineffectiveness, as well as an "incoherent mix of democratic and autocratic traits and practices."
'''

###
###
#countries
####want to plot barchart of the the European countries by NATO and EU  membership

def polityscore_cat (row):
   if (row['polityscore'] >=6 and row['polityscore'] <= 10 ) :
      return 'Democracy'
   elif (row['polityscore'] >=-5 and row['polityscore'] <= 5 )  :
      return 'Anocracy'
   elif (row['polityscore'] >=-10 and row['polityscore'] <= -6 )  :
      return 'Autocracy'   
   else :
      return 'NA'


##calculate the age of NATO countries
##data['Years_In_Nato'] = data.apply (lambda row: AGE_YEARS (row),axis=1)
data['polityscore_cat'] = data.apply (lambda row: polityscore_cat (row),axis=1)

data.columns.values
##
epp6=data['polityscore'].groupby(data['polityscore_cat']).mean()
print(epp6)


##write filter
data['incomeperperson'][(data['polityscore'] >= -12) & (data['incomeperperson'] >=0)]
## check the group by metrics
epp6=data['polityscore'].groupby(data['polityscore_cat']).mean()
print(epp6)
##calculate the mean income perpersons based on the polityscore categories
##where there is an income and polity score
epp7=data['incomeperperson'][(data['polityscore'] >= -12) & (data['incomeperperson'] >=0)].groupby(data['polityscore_cat']).mean()
print(epp7)
##"Anocracy" is a term used to describe a regime type that is characterized by inherent qualities of political instability and ineffectiveness, as well as an "incoherent mix of democratic and autocratic traits and practices.
##An autocracy is a system of government in which supreme power is concentrated in the hands of one person, whose decisions are subject to neither external legal restraints nor regularized mechanisms of popular control (except perhaps for the implicit threat of a coup d'état or mass insurrection)

# bivariate bar graph C->Q
# do not include unklnown politly scores
sns.factorplot(x='polityscore_cat', y='incomeperperson', data=data[(data['polityscore'] >= -12) & (data['incomeperperson'] >=0)], kind="bar", ci=None)
plt.xlabel('Politly Score Group')
plt.ylabel('Mean Income Per Person')
plt.title("Political category and average income per person plot")

####
####
####


sns.factorplot(x='polityscore_cat', y='armedforcesrate', data=data[(data['polityscore'] >= -12) & (data['incomeperperson'] >=0)], kind="bar", ci=None)
plt.xlabel('Politly Score Group')
plt.ylabel('Mean Armed forces rate')
plt.title("Political category and average armed forces rate plot")



####

scat3 = sns.regplot(x="armedforcesrate", y="incomeperperson", data=data)
plt.xlabel('Armed Forces rate')
plt.ylabel('Income per Person')
plt.title('Scatterplot for the Association Between Armed forces rate and Income per Person')


##With no regession
ggplot(data, aes(x='armedforcesrate', y='incomeperperson', color="NATO_EU_MEMBERSHIP")) +\
    geom_point() +\
    xlab("armed forces rate") + ylab("IncomePerPerson") + ggtitle("Gapminder")

## lets look at the data
ggplot(data, aes(x='armedforcesrate', y='incomeperperson', color="NATO_EU_MEMBERSHIP")) +\
    geom_point() +\
     geom_smooth(method='lm') +\
    xlab("armed forces rate") + ylab("IncomePerPerson") + ggtitle("Gapminder")

##now lets look at the different types of categories

ggplot(data[(data['polityscore'] >= -12) & (data['incomeperperson'] >=0)], aes(x='armedforcesrate', y='incomeperperson', color="polityscore_cat")) +\
    geom_point() +\
    xlab("armed forces rate") + ylab("IncomePerPerson") + ggtitle("Gapminder")

##with regession lines

ggplot(data[(data['polityscore'] >= -12) & (data['incomeperperson'] >=0)], aes(x='armedforcesrate', y='incomeperperson', color="polityscore_cat")) +\
    geom_point() +\
     geom_smooth(method='lm') +\
    xlab("armed forces rate") + ylab("IncomePerPerson") + ggtitle("Gapminder")

##finally lets factor with Euorpean

ggplot(data[(data['polityscore'] >= -12) & (data['incomeperperson'] >=0) & (data['European'] =='Europe')], aes(x='armedforcesrate', y='incomeperperson', color="polityscore_cat")) +\
    geom_point() +\
     geom_smooth(method='lm') +\
    facet_grid("NATO_EU_MEMBERSHIP") +\
    xlab("armed forces rate") + ylab("IncomePerPerson") + ggtitle("Gapminder")


##Regression week 1
###armedforcesrate=data['armedforcesrate'][(data['armedforcesrate'] >= 0)].values

from ggplot import ggplot, aes, geom_boxplot
##filter by europe
europe_onlyarmedforcesrate=data[(data['European'] == 'Europe')]

pandas.DataFrame.describe(europe_onlyarmedforcesrate)

pandas.DataFrame.describe(data)


ggplot(europe_onlyarmedforcesrate, aes(x='armedforcesrate', y='NATO_EU_MEMBERSHIP')) + geom_boxplot() +\
    xlab("armed forces rate") + ylab(" Nato EU membership status") + ggtitle("Boxplot for armed forces rates for gapminder Nato Eu membership for European countries")

##income per person

ggplot(europe_onlyarmedforcesrate[(europe_onlyarmedforcesrate['polityscore'] >= -12) & (europe_onlyarmedforcesrate['incomeperperson'] >=0)]
, aes(x='incomeperperson', y='NATO_EU_MEMBERSHIP')) + geom_boxplot() +\
    xlab("armed forces rate") + ylab(" Nato EU membership status") + ggtitle("Boxplot for incom per person score for gapminder Nato Eu membership for European countries")



##Regression week 2




import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn

import sklearn
from sklearn import preprocessing


scat1 = seaborn.regplot(x="armedforcesrate", y="incomeperperson", scatter=True, data=data)
plt.xlabel('armedforcesrate')
plt.ylabel('incomeperperson')
plt.title ('Scatterplot for the Association Between armedforcesrate and incomeperperson')
print(scat1)

reg1 = smf.ols('incomeperperson ~ armedforcesrate', data=data).fit()
print (reg1.summary())


reg1b = smf.ols('incomeperperson ~ polityscore', data=data).fit()
print (reg1b.summary())


reg2 = smf.ols('incomeperperson ~ armedforcesrate+polityscore', data=data).fit()
print (reg2.summary())


reg3 = smf.ols('incomeperperson ~ armedforcesrate+polityscore+NATO_EU_MEMBERSHIP', data=data).fit()
print (reg3.summary())

##select explanatory variable

data.columns.values
##only select not null values of polity  score
datav1=data[pandas.notnull(data['polityscore'])].copy(deep=True)
#carry out a check for the United Kingdom
datav1['polityscore'][(datav1['country'] == 'United Kingdom')]
##all looks ok
##now subset the datta selecting only polityscore and country
data2=datav1[['polityscore']]
##
## centre the polity score
x_centered = preprocessing.scale(data2[['polityscore']], with_mean=True, with_std=False) ##corrected this had wrong version of code had True and False in qoutes now its in correct format
##cast it as a dataframe
x_centered_df = pd.DataFrame(x_centered, columns=data2.columns)
## check the count
x_centered_df.count()
##all look sfine
##data
##data 3 is our second subset we will use to do some analysis
## it consists of ocuntry income per person and polity score
data3=datav1[['country','incomeperperson','polityscore']]
##reset the index's
data3=data3.reset_index()
del data3['index']
##

#data3['polityscore'].describe()
##concatanate once the indexs are reset
datav4 = pd.concat([data3, x_centered_df], axis=1)
##reset the column anmes to 
datav4.columns=['country','incomeperperson','polityscore','polityscore_cntred']
print(datav4)
##check the values

## build the regession model
reg1b = smf.ols('incomeperperson ~ polityscore_cntred', data=datav4).fit()
print (reg1b.summary())
##plot the residuals
scat1 = seaborn.regplot(x="polityscore_cntred", y="incomeperperson", scatter=True, data=datav4)
plt.xlabel('polityscore_cntred')
plt.ylabel('incomeperperson')
plt.title ('Scatterplot for the Association Between Income per person and the centred Polity score')
print(scat1)


###############
###Week 3
###########



data.columns.values
##only select not null values of polity  score
datamv1=data[pandas.notnull(data['polityscore'])&pandas.notnull(data['femaleemployrate'])&pandas.notnull(data['armedforcesrate'])&pandas.notnull(data['internetuserate'])].copy(deep=True)
#carry out a check for the United Kingdom
datamv1['polityscore'][(datav1['country'] == 'United Kingdom')]
##all looks ok
##now subset the datta selecting only polityscore and country
datam2=datamv1[['polityscore','femaleemployrate','armedforcesrate','internetuserate']]
##
## centre the polity score
mx_centered = preprocessing.scale(datam2[['polityscore','femaleemployrate','armedforcesrate','internetuserate']], with_mean=True, with_std=False) ##corrected this had wrong version of code had True and False in qoutes now its in correct format
##cast it as a dataframe
mx_centered_df = pd.DataFrame(mx_centered, columns=datam2.columns)
## check the count
mx_centered_df.count()
##all look sfine
##data
##data 3 is our second subset we will use to do some analysis
## it consists of ocuntry income per person and polity score
datam3=datamv1[['country','incomeperperson','polityscore','femaleemployrate','armedforcesrate','internetuserate']]
##reset the index's
datam3=datam3.reset_index()
del datam3['index']
##
#data3['polityscore'].describe()
##concatanate once the indexs are reset
datamv4 = pd.concat([datam3, mx_centered_df], axis=1)
datamv4.columns.values
'country', 'incomeperperson', 'polityscore', 'femaleemployrate',
       'armedforcesrate', 'internetuserate', 'polityscore',
       'femaleemployrate', 'armedforcesrate', 'internetuserate'

##reset the column anmes to 
datamv4.columns=['country','incomeperperson','polityscore','femaleemployrate','armedforcesrate','internetuserate','polityscore_cntred','femaleemployrate_cntred','armedforcesrate_cntred','internetuserate_cntred']
print(datamv4)
##
##now reset the country column as an index
datamv4reidx=datamv4.set_index('country')
datamv4reidx.columns.values
##build rergession model using the centred 
## build the multi value regession model
mreg1b = smf.ols('incomeperperson ~ polityscore_cntred+femaleemployrate_cntred+armedforcesrate_cntred+internetuserate_cntred', data=datamv4).fit()
print (mreg1b.summary())




#Q-Q plot for normality


fig4=sm.qqplot(mreg1b.resid, line='r')

# simple plot of residuals
stdres=pandas.DataFrame(mreg1b.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
plt.title('Standardized Residual Plot')


# additional regression diagnostic plots
import matplotlib.pyplot as pltt
pd.set_option('display.mpl_style', 'default')
fig2 = pltt.figure()
fig2 = sm.graphics.plot_regress_exog(mreg1b,  "polityscore_cntred",fig=fig2)
print(fig2)


##
pd.set_option('display.mpl_style', 'default')
fig = pltt.figure()
fig = sm.graphics.plot_regress_exog(mreg1b,  "internetuserate_cntred",fig=fig)
print(fig)

internetuserate_cntred 

# leverage plot
fig3=sm.graphics.influence_plot(mreg1b, size=8)
print(fig3)

##change a column to index
##you can see the countries casuig rpoblems










