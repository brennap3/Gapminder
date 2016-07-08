# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 17:10:23 2016

@author: Peter
"""

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
   elif row['country'] == 'Slovak Republic' :    
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
      return 'Not_In_Europe'

data['European'] = data.apply (lambda row: EUROPEAN (row),axis=1)

##check it worked

##'''
##Out[24]: 
##Europe            45
##Not-In-Europe    168
##dtype: int64
##'''

data['European'].value_counts(sort=False, dropna=False)

data['country']

##checked working 

def African (row):
   if row['country'] == 'Algeria' :
      return 'Africa'
   elif row['country'] == 'Angola' :
       return 'Africa'
   elif row['country'] == 'Benin' :
       return 'Africa'
   elif row['country'] == 'Botswana' :
       return 'Africa'
   elif row['country'] == 'Burkina Faso' : 
       return 'Africa'
   elif row['country'] == 'Burundi' :
       return 'Africa'
   elif row['country'] == 'Cameroon' :
       return 'Africa'
   elif row['country'] == 'Cape Verde' :
       return 'Africa'
   elif row['country'] == 'Central African Republic' :
       return 'Africa'
   elif row['country'] == 'Chad' :
       return 'Africa'
   elif row['country'] == 'Comoros' :
       return 'Africa'
   elif row['country'] == 'Congo, Dem. Rep.' :
       return 'Africa'    
   elif row['country'] == 'Congo, Rep.' :
       return 'Africa'
   elif row['country'] == 'Djibouti' :
       return 'Africa'
   elif row['country'] == 'Equatorial Guinea' :
       return 'Africa'      
   elif row['country'] == 'Eritrea' :
       return 'Africa'  
   elif row['country'] == 'Ethiopia' :
       return 'Africa'    
   elif row['country'] == 'Egypt' :
       return 'Africa'    		  
   elif row['country'] == 'Gabon' :
       return 'Africa'  
   elif row['country'] == 'Gambia' :
       return 'Africa'    
   elif row['country'] == 'Ghana' :
       return 'Africa'    
   elif row['country'] == "Cote d'Ivoire":
       return 'Africa'
   elif row['country'] == "Guinea-Bissau":
       return 'Africa'
   elif row['country'] == "Guinea":
       return 'Africa'
   elif row['country'] == "Kenya":
       return 'Africa'    
   elif row['country'] == "Lesotho":
       return 'Africa'    
   elif row['country'] == "Liberia":
       return 'Africa'    
   elif row['country'] == "Libya":
       return 'Africa'         
   elif row['country'] == "Madagascar":
       return 'Africa'    
   elif row['country'] == "Malawi":
       return 'Africa'    
   elif row['country'] == "Mali":
       return 'Africa'        
   elif row['country'] == "Mauritania":
       return 'Africa'   
   elif row['country'] == "Mauritius":
       return 'Africa'   
   elif row['country'] == "Morocco":
       return 'Africa'   
   elif row['country'] == "Mozambique":
       return 'Africa'   
   elif row['country'] == "Namibia":
       return 'Africa'       
   elif row['country'] == "Niger":
       return 'Africa'         
   elif row['country'] == "Nigeria":
       return 'Africa' 
   elif row['country'] == "Rwanda":
       return 'Africa' 
   elif row['country'] == 'Sao Tome and Principe':
       return 'Africa'
   elif row['country'] == 'Senegal':
       return 'Africa'    
   elif row['country'] == 'Seychelles':
       return 'Africa'            
   elif row['country'] == 'Sierra Leone':
       return 'Africa'            
   elif row['country'] == 'Somalia':
       return 'Africa'        
   elif row['country'] == 'South Sudan':
       return 'Africa'    
   elif row['country'] == 'South Africa':
       return 'Africa'    
   elif row['country'] == 'Sudan':
       return 'Africa'    
   elif row['country'] == 'Swaziland':
       return 'Africa'     
   elif row['country'] == 'Tanzania':
       return 'Africa'     
   elif row['country'] == 'Togo':
       return 'Africa'    
   elif row['country'] == 'Tunisia':
       return 'Africa'    
   elif row['country'] == 'Uganda':
       return 'Africa'    
   elif row['country'] == 'Zambia':
       return 'Africa'    
   elif row['country'] == 'Zimbabwe':
       return 'Africa'    
   elif row['country'] == 'Somaliland':
       return 'Africa'    
   else :
       return 'Not_In_Africa'

data['African'] = data.apply (lambda row: African (row),axis=1)     

data['African'].value_counts(sort=False, dropna=False)
     
def Asian (row):
   if row['country'] == 'Afganistan' :
      return 'Asia'
   elif row['country'] == 'Armenia' :
       return 'Asia'
   elif row['country'] == 'Bahrain' :
       return 'Asia'       
   elif row['country'] == 'Bangladesh' :
       return 'Asia'       
   elif row['country'] == 'Bhutan' :
       return 'Asia'       
   elif row['country'] == 'Brunei' :
       return 'Asia'       
   elif row['country'] == 'Cambodia' :
       return 'Asia'       
   elif row['country'] == 'China' :
       return 'Asia'
   elif row['country'] == 'Georgia' :
       return 'Asia'
   elif row['country'] == 'India' :
       return 'Asia'
   elif row['country'] == 'Iran' :
       return 'Asia'	   
   elif row['country'] == 'Indonesia' :
       return 'Asia'	   
   elif row['country'] == 'Iraq' :
       return 'Asia'	   
   elif row['country'] == 'Israel' :
       return 'Asia'	   
   elif row['country'] == 'Japan' :
       return 'Asia'	   
   elif row['country'] == 'Jordan' :
       return 'Asia'	   
   elif row['country'] == 'Kazakhstan' :
       return 'Asia'	   
   elif row['country'] == 'Korea, Dem. Rep.' :
       return 'Asia'	   
   elif row['country'] == 'Korea, Rep.' :
       return 'Asia'	   	   
   elif row['country'] == 'Kuwait' :
       return 'Asia'	   	   
   elif row['country'] == 'Kyrgyzstan' :
       return 'Asia'	   	  	
   elif row['country'] == 'Laos' :
       return 'Asia'	
   elif row['country'] == 'Lebanon' :
       return 'Asia'	
   elif row['country'] == 'Malaysia' :
       return 'Asia'	
   elif row['country'] == 'Maldives' :
       return 'Asia'		   
   elif row['country'] == 'Mongolia' :
       return 'Asia'	
   elif row['country'] == 'Myanmar' :
       return 'Asia'	
   elif row['country'] == 'Nepal' :
       return 'Asia'	
   elif row['country'] == 'Oman' :
       return 'Asia'		 
   elif row['country'] == 'Pakistan' :
       return 'Asia'		 
   elif row['country'] == 'Philippines' :
       return 'Asia'		 
   elif row['country'] == 'Qatar' :
       return 'Asia'		 
   elif row['country'] == 'Saudi Arabia' :
       return 'Asia'		 
   elif row['country'] == 'Singapore' :
       return 'Asia'		 
   elif row['country'] == 'Sri Lanka' :
       return 'Asia'		 
   elif row['country'] == 'Syria' :
       return 'Asia'		 	   
   elif row['country'] == 'Tajikistan' :
       return 'Asia'		
   elif row['country'] == 'Thailand' :
       return 'Asia'		
   elif row['country'] == 'Timor-Leste' :
       return 'Asia'		
   elif row['country'] == 'Turkey' :
       return 'Asia'		
   elif row['country'] == 'Turkmenistan' :
       return 'Asia'	   
   elif row['country'] == 'United Arab Emirates' :
       return 'Asia'	   
   elif row['country'] == 'Uzbekistan' :
       return 'Asia'	   
   elif row['country'] == 'Vietnam' :
       return 'Asia'	   	   
   elif row['country'] == 'Yemen' :
       return 'Asia'	   	      
   else :
      return 'Not_In_Asia'

data['Asian'] = data.apply (lambda row: Asian (row),axis=1)  

data['Asian'].value_counts(sort=False, dropna=False)   

##data[['country','incomeperperson','polityscore_cat']][(data.European=='Europe') & (data.polityscore_cat!='NA') ]

##data[(data.Asian=='Asian')]
      
      
def Mid_East (row):
  if   row['country'] == 'Bahrain' :
       return 'Middle_East'
  elif row['country'] == 'Cyprus' :
       return 'Middle_East'
  elif row['country'] == 'Egypt' :
       return 'Middle_East'     
  elif row['country'] == 'Iran' :
       return 'Middle_East' 	   
  elif row['country'] == 'Iraq' :
       return 'Middle_East'	   
  elif row['country'] == 'Israel' :
       return 'Middle_East'	   
  elif row['country'] == 'Jordan' :
       return 'Middle_East'	   	   
  elif row['country'] == 'Kuwait' :
       return 'Middle_East'	   	   
  elif row['country'] == 'Lebanon' :
       return 'Middle_East'	
  elif row['country'] == 'Oman' :
       return 'Middle_East'		 
  elif row['country'] == 'Qatar' :
       return 'Middle_East'		 
  elif row['country'] == 'Saudi Arabia' :
       return 'Middle_East'		 
  elif row['country'] == 'Syria' :
       return 'Middle_East'		 	   
  elif row['country'] == 'Turkey' :
       return 'Middle_East'		
  elif row['country'] == 'United Arab Emirates' :
       return 'Middle_East'	   
  elif row['country'] == 'Yemen' :
       return 'Middle_East'	   	      
  else :
      return 'Not_In_Middle_East'     
    
data['Mid_East'] = data.apply (lambda row: Mid_East (row),axis=1)  

data['Mid_East'].value_counts(sort=False, dropna=False)  	

def North_American (row):
  if   row['country'] == 'Antigua and Barbuda' :
       return 'North_America'       
  elif row['country'] == 'Bahamas' :
       return 'North_America'        
  elif row['country'] == 'Barbados' :
       return 'North_America'       
  elif row['country'] == 'Belize' :
       return 'North_America'             
  elif row['country'] == 'Canada' :
       return 'North_America'           
  elif row['country'] == 'Costa Rica' :
       return 'North_America'        
  elif row['country'] == 'Cuba' :
       return 'North_America'    
  elif row['country'] == 'Dominica' :
       return 'North_America'      
  elif row['country'] == 'Dominican Republic' :
       return 'North_America'          
  elif row['country'] == 'El Salvador' :
       return 'North_America'
  elif row['country'] == 'Grenada' :
       return 'North_America'
  elif row['country'] == 'Guatemala' :
       return 'North_America'
  elif row['country'] == 'Haiti' :
       return 'North_America'	   
  elif row['country'] == 'Honduras' :
       return 'North_America'	  
  elif row['country'] == 'Jamaica' :
       return 'North_America'	 
  elif row['country'] == 'Mexico' :
       return 'North_America'	
  elif row['country'] == 'Nicaragua' :
       return 'North_America'	
  elif row['country'] == 'Panama' :
       return 'North_America'	
  elif row['country'] == 'Panama' :
       return 'North_America'	
  elif row['country'] == 'Saint Kitts and Nevis' :
       return 'North_America'	
  elif row['country'] == 'Saint Lucia' :
       return 'North_America'	
  elif row['country'] == 'Saint Vincent and the Grenadines' :
       return 'North_America'	
  elif row['country'] == 'Trinidad and Tobago' :
       return 'North_America'		   
  elif row['country'] == 'United States' :
       return 'North_America'		   
  else :
      return 'Not_In_North_America'     

data['North_American'] = data.apply (lambda row: North_American (row),axis=1)  

data['North_American'].value_counts(sort=False, dropna=False)  	

	   
def Carribean_Central_America (row):
   if   row['country'] == 'Antigua and Barbuda' :
       return 'Carribean_Central_American'       
   elif row['country'] == 'Bahamas' :
       return 'Carribean_Central_American'        
   elif row['country'] == 'Barbados' :
       return 'Carribean_Central_American'       
   elif row['country'] == 'Belize' :
       return 'Carribean_Central_American'             
   elif row['country'] == 'Costa Rica' :
       return 'Carribean_Central_American'        
   elif row['country'] == 'Cuba' :
       return 'Carribean_Central_American'    
   elif row['country'] == 'Dominica' :
       return 'Carribean_Central_American'      
   elif row['country'] == 'Dominican Republic' :
       return 'Carribean_Central_American'          
   elif row['country'] == 'El Salvador' :
       return 'Carribean_Central_American'
   elif row['country'] == 'Grenada' :
       return 'Carribean_Central_American'
   elif row['country'] == 'Guatemala' :
       return 'Carribean_Central_American'
   elif row['country'] == 'Haiti' :
       return 'Carribean_Central_American'	   
   elif row['country'] == 'Honduras' :
       return 'Carribean_Central_American'	  
   elif row['country'] == 'Jamaica' :
       return 'Carribean_Central_American'	 
   elif row['country'] == 'Nicaragua' :
       return 'Carribean_Central_American'	
   elif row['country'] == 'Panama' :
       return 'Carribean_Central_American'	
   elif row['country'] == 'Saint Kitts and Nevis' :
       return 'Carribean_Central_American'	
   elif row['country'] == 'Saint Lucia' :
       return 'Carribean_Central_American'	
   elif row['country'] == 'Saint Vincent and the Grenadines' :
       return 'Carribean_Central_American'	
   elif row['country'] == 'Trinidad and Tobago' :
       return 'Carribean_Central_American'		   
   else :
      return 'Not_In_Carribean_Central_American'     
	   
data['Carribean_Central_America'] = data.apply (lambda row: Carribean_Central_America (row),axis=1)  

data['Carribean_Central_America'].value_counts(sort=False, dropna=False)  	


	   
##Algeria, Angola, Ecuador, Iran, Iraq, Kuwait, Libya, Nigeria, Qatar, Saudi Arabia, United Arab Emirates and Venezuela	   
	   
	   
def OPEC (row):
   if   row['country'] == 'Algeria' :
       return 'OPEC_MEMBER'       
   elif row['country'] == 'Angola' :
       return 'OPEC_MEMBER'        
   elif row['country'] == 'Ecuador' :
       return 'OPEC_MEMBER'       
   elif row['country'] == 'Iran' :
       return 'OPEC_MEMBER'             
   elif row['country'] == 'Iraq' :
       return 'OPEC_MEMBER'        
   elif row['country'] == 'Kuwait' :
       return 'OPEC_MEMBER'    
   elif row['country'] == 'Libya' :
       return 'OPEC_MEMBER'      
   elif row['country'] == 'Nigeria' :
       return 'OPEC_MEMBER'          
   elif row['country'] == 'Qatar' :
       return 'OPEC_MEMBER'
   elif row['country'] == 'Saudi Arabia' :
       return 'OPEC_MEMBER'
   elif row['country'] == 'United Arab Emirates' :
       return 'OPEC_MEMBER'
   elif row['country'] == 'Venezuela' :
       return 'OPEC_MEMBER'	   
   else :
      return 'Not_In_OPEC'  

data['OPEC'] = data.apply (lambda row: OPEC (row),axis=1)  

data['OPEC'].value_counts(sort=False, dropna=False)  	
	  
def Arab_League (row):
   if   row['country'] == 'Algeria' :
       return 'Arab_League_MEMBER'       
   elif row['country'] == 'Bahrain' :
       return 'Arab_League_MEMBER'        
   elif row['country'] == 'Comoros' :
       return 'Arab_League_MEMBER'       
   elif row['country'] == 'Djibouti' :
       return 'Arab_League_MEMBER'             
   elif row['country'] == 'Egypt' :
       return 'Arab_League_MEMBER'        
   elif row['country'] == 'Iraq' :
       return 'Arab_League_MEMBER'    
   elif row['country'] == 'Jordan' :
       return 'Arab_League_MEMBER'      
   elif row['country'] == 'Kuwait' :
       return 'Arab_League_MEMBER'          
   elif row['country'] == 'Lebanon' :
       return 'Arab_League_MEMBER'
   elif row['country'] == 'Libya' :
       return 'Arab_League_MEMBER'
   elif row['country'] == 'Mauritania' :
       return 'Arab_League_MEMBER'
   elif row['country'] == 'Morocoo' :
       return 'Arab_League_MEMBER'	   
   elif row['country'] == 'Oman' :
       return 'Arab_League_MEMBER'
   elif row['country'] == 'West Bank and Gaza' :
       return 'Arab_League_MEMBER'
   elif row['country'] == 'Qatar' :
       return 'Arab_League_MEMBER'
   elif row['country'] == 'Saudi Arabia' :
       return 'Arab_League_MEMBER'
   elif row['country'] == 'Somalia' :
       return 'Arab_League_MEMBER'
   elif row['country'] == 'Sudan' :
       return 'Arab_League_MEMBER'
   elif row['country'] == 'Syria' :
       return 'Arab_League_MEMBER'	   
   elif row['country'] == 'Tunisia' :
       return 'Arab_League_MEMBER'	   	 
   elif row['country'] == 'United Arab Emirates' :
       return 'Arab_League_MEMBER'	   	 
   elif row['country'] == 'Yemen' :
       return 'Arab_League_MEMBER'	   
   elif row['country'] == 'Eritrea' :
       return 'Arab_League_MEMBER'	
   else :
      return 'Not_In_Arab_League'  
##
      
data['Arab_League'] = data.apply (lambda row: Arab_League (row),axis=1)  

data['Arab_League'].value_counts(sort=False, dropna=False)  	
      
##ASEAN is a regional grouping with security, economic and social aspects	  
	  
def ASEAN_ARF (row):
   if row['country'] == 'Australia' :
      return 'ASEAN_ARF_MEMBER'
   elif row['country'] == 'Bangladesh' :
       return 'ASEAN_ARF_MEMBER'
   elif row['country'] == 'Brunei' :
       return 'ASEAN_ARF_MEMBER'       
   elif row['country'] == 'Cambodia' :
       return 'ASEAN_ARF_MEMBER'       
   elif row['country'] == 'Canada' :
       return 'ASEAN_ARF_MEMBER'       
   elif row['country'] == 'China' :
       return 'ASEAN_ARF_MEMBER'       
   elif row['country'] == 'India' :
       return 'ASEAN_ARF_MEMBER'       
   elif row['country'] == 'Indonesia' :
       return 'ASEAN_ARF_MEMBER'
   elif row['country'] == 'Japan' :
       return 'ASEAN_ARF_MEMBER'
   elif row['country'] == 'Korea, Dem. Rep.' :
       return 'ASEAN_ARF_MEMBER'
   elif row['country'] == 'Korea, Rep.' :
       return 'ASEAN_ARF_MEMBER'	   
   elif row['country'] == 'Laos' :
       return 'ASEAN_ARF_MEMBER'	   
   elif row['country'] == 'Malaysia' :
       return 'ASEAN_ARF_MEMBER'	   
   elif row['country'] == 'Myanmar' :
       return 'ASEAN_ARF_MEMBER'	   
   elif row['country'] == 'Mongolia' :
       return 'ASEAN_ARF_MEMBER'	   
   elif row['country'] == 'New Zealand' :
       return 'ASEAN_ARF_MEMBER'	   
   elif row['country'] == 'Pakistan' :
       return 'ASEAN_ARF_MEMBER'	   
   elif row['country'] == 'Papua New Guinea' :
       return 'ASEAN_ARF_MEMBER'	   
   elif row['country'] == 'Phillipines' :
       return 'ASEAN_ARF_MEMBER'	   	   
   elif row['country'] == 'Russian Federation' :
       return 'ASEAN_ARF_MEMBER'	   	   
   elif row['country'] == 'Singapore' :
       return 'ASEAN_ARF_MEMBER'	   	  	
   elif row['country'] == 'Sri Lanka' :
       return 'ASEAN_ARF_MEMBER'	
   elif row['country'] == 'Thailand' :
       return 'ASEAN_ARF_MEMBER'	
   elif row['country'] == 'Timor-Leste' :
       return 'ASEAN_ARF_MEMBER'	
   elif row['country'] == 'United States' :
       return 'ASEAN_ARF_MEMBER'		   
   elif row['country'] == 'Vietnam' :
       return 'ASEAN_ARF_MEMBER'	
   else :
      return 'Not_In_ASEAN_ARF'

data['ASEAN_ARF'] = data.apply (lambda row: ASEAN_ARF (row),axis=1)  

data['ASEAN_ARF'].value_counts(sort=False, dropna=False)  	
 	  

def South_American (row):
   if row['country'] == 'Argentina' :
      return 'South_America'
   elif row['country'] == 'Bolivia' :
       return 'South_America'
   elif row['country'] == 'Brazil' :
       return 'South_America'       
   elif row['country'] == 'Chile' :
       return 'South_America'       
   elif row['country'] == 'Colombia' :
       return 'South_America'       
   elif row['country'] == 'Ecuador' :
       return 'South_America'       
   elif row['country'] == 'Guyana' :
       return 'South_America'       
   elif row['country'] == 'Paraguay' :
       return 'South_America'
   elif row['country'] == 'Peru' :
       return 'South_America'
   elif row['country'] == 'Suriname' :
       return 'South_America'
   elif row['country'] == 'Uruguay' :
       return 'South_America'	   
   elif row['country'] == 'Venezuala' :
       return 'South_America'	   
   else :
      return 'Not_South_America'


data['South_American'] = data.apply (lambda row: South_American (row),axis=1)  

data['South_American'].value_counts(sort=False, dropna=False)  	

##	

###http://www.nato.int/cps/en/natohq/topics_52044.htm

##NATO data

Nato_Countries = pandas.DataFrame({ 'country' : ('Albania','Belgium','Bulgaria','Canada','Croatia','Czech Republic','Denmark','Estonia','France','Germany','Greece','Hungary','Iceland','Italy','Latvia','Lithuania','Luxembourg','Netherlands','Norway','Poland','Portugal','Romania','Slovak Republic','Slovenia','Spain','Turkey','United Kingdom','United States'),
                     'Year_Joined' : (2009,1949,2004,1949,2009,1999,1949,2004,1949,1955,1952,1999,1949,1949,2004,2004,1949,1949,1949,1999,1949,2004,2004,2004,1982,1952,1949,1949),
                     'Is_Nato_Country' : 'Nato_Member'
                        })

##Enhanced data join NATO data

data.columns.values 


data=pandas.merge(data, Nato_Countries,how='left',on='country')

##data.columns.values
##check that all column values have been added


data['Is_Nato_Country']=data['Is_Nato_Country'].fillna('Not_in_Nato')

##

data['Is_Nato_Country'].value_counts(sort=False, dropna=False)  




data.columns.values
##year joined needs to be renamed
data.rename(columns={'Year_Joined': 'Year_Joined_Nato'}, inplace=True)
## change columns names
data.columns.values

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
      return 'Not_In_EU'


data['Eu_Member'] = data.apply (lambda row: EUMEMBER (row),axis=1)	   

data['Eu_Member'].value_counts(sort=False, dropna=False)  

data.columns.values 	   

import time
##check how to calc time
print (time.strftime("%Y"))
##write unction to calculate the  age of NATO countries based on the current date
def AGE_YEARS (row):
   current_year=time.strftime("%Y")
   if row['Year_Joined_Nato'] >0 :
      return (int(current_year)-int(row['Year_Joined_Nato']))
   else :
      return 0

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

data.columns.values 

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


##array(['country', 'incomeperperson', 'alcconsumption', 'armedforcesrate',
##       'breastcancerper100th', 'co2emissions', 'femaleemployrate',
##       'hivrate', 'internetuserate', 'lifeexpectancy', 'oilperperson',
##       'polityscore', 'relectricperperson', 'suicideper100th',
##       'employrate', 'urbanrate', 'categories', 'European', 'African',
##       'Asian', 'Mid_East', 'North_American', 'Carribean_Central_America',
##       'OPEC', 'Arab_League', 'ASEAN_ARF', 'South_American',
##       'Is_Nato_Country', 'Year_Joined_Nato', 'Eu_Member', 'Years_In_Nato',
##       'NATO_EU_MEMBERSHIP', 'polityscore_cat'], dtype=object)
 
##data = data.drop('Is_Nato_Country_y', 1)
##data.rename(columns={'Is_Nato_Country_x': 'Is_Nato_Country'}, inplace=True)

data['European'].value_counts(sort=False, dropna=False)  	

data['European'].replace("Europe",1,inplace=True)  
data['European'].replace("Not_In_Europe",0,inplace=True)

data['African'].value_counts(sort=False, dropna=False)  	

data['African'].replace("Africa",1,inplace=True)  
data['African'].replace("Not_In_Africa",0,inplace=True)

data['African'].value_counts(sort=False, dropna=False)  	

data['Asian'].value_counts(sort=False, dropna=False)

data['Asian'].replace("Asia",1,inplace=True)  
data['Asian'].replace("Not_In_Asia",0,inplace=True)

data['Asian'].value_counts(sort=False, dropna=False)


##'Mid_East'

data['Mid_East'].value_counts(sort=False, dropna=False)

data['Mid_East'].replace("Middle_East",1,inplace=True)  
data['Mid_East'].replace("Not_In_Middle_East",0,inplace=True)

data['Mid_East'].value_counts(sort=False, dropna=False)

data['North_American'].value_counts(sort=False, dropna=False)

data['North_American'].replace("North_America",1,inplace=True)  
data['North_American'].replace("Not_In_North_America",0,inplace=True)

data['North_American'].value_counts(sort=False, dropna=False)

data['Carribean_Central_America'].value_counts(sort=False, dropna=False)

data['Carribean_Central_America'].replace("Carribean_Central_American",1,inplace=True)  
data['Carribean_Central_America'].replace("Not_In_Carribean_Central_American",0,inplace=True)

data['Carribean_Central_America'].value_counts(sort=False, dropna=False)

data['OPEC'].replace("OPEC_MEMBER",1,inplace=True)  
data['OPEC'].replace("Not_In_OPEC",0,inplace=True)

data['OPEC'].value_counts(sort=False, dropna=False)

data['Arab_League'].value_counts(sort=False, dropna=False)

data['Arab_League'].replace("Not_In_Arab_League",0,inplace=True)  
data['Arab_League'].replace("Arab_League_MEMBER",1,inplace=True)

data['Arab_League'].value_counts(sort=False, dropna=False)

##'ASEAN_ARF'

data['ASEAN_ARF'].value_counts(sort=False, dropna=False)

data['ASEAN_ARF'].replace("Not_In_ASEAN_ARF",0,inplace=True)  
data['ASEAN_ARF'].replace("ASEAN_ARF_MEMBER",1,inplace=True)

data['ASEAN_ARF'].value_counts(sort=False, dropna=False)

##'South_American'

data['South_American'].value_counts(sort=False, dropna=False)

data['South_American'].replace("Not_South_America",0,inplace=True)  
data['South_American'].replace("South_America",1,inplace=True)

data['South_American'].value_counts(sort=False, dropna=False)

##'Is_Nato_Country'

data['Is_Nato_Country'].value_counts(sort=False, dropna=False)

data['Is_Nato_Country'].replace("Not_in_Nato",0,inplace=True)  
data['Is_Nato_Country'].replace("Nato_Member",1,inplace=True)

data['Is_Nato_Country'].value_counts(sort=False, dropna=False)

##'Eu_Member'

data['Eu_Member'].value_counts(sort=False, dropna=False)

data['Eu_Member'].replace("Not_In_EU",0,inplace=True)  
data['Eu_Member'].replace("EU",1,inplace=True)

data['Eu_Member'].value_counts(sort=False, dropna=False)


##'polityscore_cat'

data['polityscore_cat'].value_counts(sort=False, dropna=False)

data['polityscore_cat'].replace("Anocracy",0,inplace=True)
data['polityscore_cat'].replace("Autocracy",0,inplace=True)
data['polityscore_cat'].replace("NA",0,inplace=True)  
data['polityscore_cat'].replace("Democracy",1,inplace=True)

data['polityscore_cat'].value_counts(sort=False, dropna=False)




  
##
  ###
  ###
##
##lets try   



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


data2=datafullset[['incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  'European', 'African',
         'Asian', 'Mid_East', 'North_American', 'Carribean_Central_America',
         'OPEC', 'Arab_League', 'ASEAN_ARF', 'South_American', 'Eu_Member','Is_Nato_Country','Years_In_Nato',
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide',
         'polityscore']]

data2.count

data_clean2 = data2.dropna() ## drop all na values cant handle nulls

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
data_clean2['European']=preprocessing.scale(data_clean2['European'].astype('float64'))
data_clean2['African']=preprocessing.scale(data_clean2['African'].astype('float64'))
data_clean2['Asian']=preprocessing.scale(data_clean2['Asian'].astype('float64'))
data_clean2['Mid_East']=preprocessing.scale(data_clean2['Mid_East'].astype('float64'))
data_clean2['North_American']=preprocessing.scale(data_clean2['North_American'].astype('float64'))
data_clean2['Carribean_Central_America']=preprocessing.scale(data_clean2['Carribean_Central_America'].astype('float64'))
data_clean2['OPEC']=preprocessing.scale(data_clean2['OPEC'].astype('float64'))
data_clean2['Arab_League']=preprocessing.scale(data_clean2['Arab_League'].astype('float64'))
data_clean2['ASEAN_ARF']=preprocessing.scale(data_clean2['ASEAN_ARF'].astype('float64'))
data_clean2['South_American']=preprocessing.scale(data_clean2['South_American'].astype('float64'))
data_clean2['Eu_Member']=preprocessing.scale(data_clean2['Eu_Member'].astype('float64'))
data_clean2['Is_Nato_Country']=preprocessing.scale(data_clean2['Is_Nato_Country'].astype('float64'))
data_clean2['Years_In_Nato']=preprocessing.scale(data_clean2['Years_In_Nato'].astype('float64'))
data_clean2['CPI2015']=preprocessing.scale(data_clean2['CPI2015'].astype('float64'))
data_clean2['World Economic Forum EOS']=preprocessing.scale(data_clean2['World Economic Forum EOS'].astype('float64'))
data_clean2['PRS International Country Risk Guide']=preprocessing.scale(data_clean2['PRS International Country Risk Guide'].astype('float64'))
##

###
###check the standardization worked

target = data_clean2.polityscore
 
# standardize predictors to have mean=0 and sd=1
predictors=data_clean2[['incomeperperson', 'alcconsumption', 'armedforcesrate',
         'femaleemployrate',
         'internetuserate', 'lifeexpectancy', 
         'employrate',  'European', 'African',
         'Asian', 'Mid_East', 'North_American', 'Carribean_Central_America',
         'OPEC', 'Arab_League', 'ASEAN_ARF', 'South_American', 'Eu_Member','Is_Nato_Country','Years_In_Nato',
         'CPI2015','World Economic Forum EOS','PRS International Country Risk Guide']]



###
###

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=123)
                                                              

import sklearn.linear_model   
from sklearn.linear_model import Ridge                                                           
# specify the lasso regression model
# change cv to 5
##model=sklearn.linear_model.LassoLarsCV(cv=5, precompute=False).fit(pred_train,tar_train)

clf = Ridge(alpha=1.0)
clf.fit(predictors, target) 


# print variable names and regression coefficients
dict(zip(predictors.columns, clf.coef_))

##{'ASEAN_ARF': 0.43410496413132893,
## 'African': -1.5303218680794426,
## 'Arab_League': -1.3108673329035154,
## 'Asian': -0.91413153833088934,
## 'CPI2015': 4.0195090042528676,
## 'Carribean_Central_America': 0.56153883159931162,
## 'Eu_Member': 0.40039457507880677,
## 'European': -0.41316328504220984,
## 'Is_Nato_Country': 0.98948429199508237,
## 'Mid_East': 0.17053012546651281,
## 'North_American': 0.084753680917910038,
## 'OPEC': -1.1089629319743224,
## 'PRS International Country Risk Guide': 0.031964147454483317,
## 'South_American': 0.72312771797922715,
## 'World Economic Forum EOS': -2.8644209286726774,
## 'Years_In_Nato': -0.51606419959818395,
## 'alcconsumption': 0.054933365499560412,
## 'armedforcesrate': -0.77442113472774921,
## 'employrate': -2.2402272226248297,
## 'femaleemployrate': 0.63282401088709272,
## 'incomeperperson': 1.4203818964664965,
## 'internetuserate': -1.5018973416359096,
## 'lifeexpectancy': -0.55281204171334886}


clf2 = Ridge(alpha=0.1)
clf2.fit(predictors, target)
dict(zip(predictors.columns, clf2.coef_))

## {'ASEAN_ARF': 0.43979869700447427,
##  'African': -1.7356059610544605,
##  'Arab_League': -1.2474908078192191,
##  'Asian': -0.95887001945621142,
##  'CPI2015': 5.3908592760150444,
##  'Carribean_Central_America': 0.63501788045253371,
##  'Eu_Member': 0.3034845863900788,
##  'European': -0.38896903857405574,
##  'Is_Nato_Country': 1.0783846658518017,
##  'Mid_East': 0.21037230183069561,
##  'North_American': -0.083845123985918993,
##  'OPEC': -1.1202813801581388,
##  'PRS International Country Risk Guide': -0.74725225492227276,
##  'South_American': 0.66236671226827892,
##  'World Economic Forum EOS': -3.3615427709751486,
##  'Years_In_Nato': -0.5711710215848389,
##  'alcconsumption': -0.022315767551261911,
##  'armedforcesrate': -0.85512858304864769,
##  'employrate': -2.2735733349356,
##  'femaleemployrate': 0.64785408566657765,
##  'incomeperperson': 1.5344769799784839,
##  'internetuserate': -1.7878108586327877,
##  'lifeexpectancy': -0.59803994190501053}



clf3 = Ridge(alpha=0.6)
clf3.fit(predictors, target)
dict(zip(predictors.columns, clf3.coef_))

## Out[91]: 
## {'ASEAN_ARF': 0.43592792788542872,
##  'African': -1.6124721173568941,
##  'Arab_League': -1.2849397577417903,
##  'Asian': -0.93158599770496497,
##  'CPI2015': 4.5067139491969623,
##  'Carribean_Central_America': 0.58797191724689635,
##  'Eu_Member': 0.36486995661876764,
##  'European': -0.40988452793667696,
##  'Is_Nato_Country': 1.0275970125659208,
##  'Mid_East': 0.18789263349967222,
##  'North_American': 0.023842089233059895,
##  'OPEC': -1.1124140238963223,
##  'PRS International Country Risk Guide': -0.23497944934544723,
##  'South_American': 0.70066482199431113,
##  'World Economic Forum EOS': -3.0537150620326239,
##  'Years_In_Nato': -0.54147238366778694,
##  'alcconsumption': 0.025955306179350299,
##  'armedforcesrate': -0.80490997434476219,
##  'employrate': -2.2668858064060369,
##  'femaleemployrate': 0.65168313557767943,
##  'incomeperperson': 1.4715961761338343,
##  'internetuserate': -1.6093255293145361,
##  'lifeexpectancy': -0.57433108305052583}
## In [92]:

from sklearn.metrics import mean_squared_error
from sklearn import linear_model


##

##lets use a simple grid search first to tune this bitch


model=sklearn.linear_model.RidgeCV(cv=5, alphas=[0.0001,0.001,0.1, 1.0, 10.0,100]).fit(pred_train,tar_train)

dict(zip(predictors.columns, model.coef_))

model.alpha_
model.coef_
model.alphas
model.get_params(y)



##try the same with  CV


alphas = [100,10,1,0.1,0.01,0.001,0.0001]
clf = sklearn.linear_model.Ridge(fit_intercept=False)
errors = []
coefs = []
#
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(pred_train, tar_train)
    coefs.append(clf.coef_)
    errors.append(mean_squared_error(tar_train, clf.predict(pred_train)))

# Display results

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.axvline(np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')

plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()


#

ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.axvline(np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Coefficient error as a function of the regularization')
plt.axis('tight')

plt.show()





