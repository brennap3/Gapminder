
# coding: utf-8

# In[1]:

import plotly.plotly as py
from plotly.graph_objs import *

trace0 = Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17]
)
trace1 = Scatter(
    x=[1, 2, 3, 4],
    y=[16, 5, 11, 9]
)
data = Data([trace0, trace1])

unique_url = py.plot(data, filename = 'basic-line')

##boiler plate ignore


# In[2]:

##change directory
#C:\Python27\ProgrammingAssignment1Data\ProgrammingAssignment1Data\Data

#C:\Python27>python -c "import plotly; plotly.tools.set_credentials_file(username
#='brennap3', api_key='d373fc7ksj')"

import os
filepath='C:/Python27/ProgrammingAssignment1Data/ProgrammingAssignment1Data/Data' 
os.chdir(filepath)


# In[3]:

##read in files
import pandas as pd
ddf1=pd.read_csv('ExcelFormattedGISTEMPData.csv',header=0)


# In[4]:

ddf1


# In[5]:

ddf2=pd.read_csv('ExcelFormattedGISTEMPData2.csv',header=0)


# In[6]:

ddf2


# In[7]:

ddf2.Year


# In[8]:

import cufflinks as cf


# In[9]:

help(ddf1.iplot)


# In[10]:

xd=ddf2[['Year','Glob','NHem','SHem']]
ddf2v0 = xd.set_index('Year')


# In[11]:

ddf2v0.columns = ['Global temperature change', 'Northern hemisphere temperature change', 'Southern hemisphere temperature change']


# In[12]:

ddf2v0


# In[13]:

ddf2v0[['Global temperature change', 'Northern hemisphere temperature change', 'Southern hemisphere temperature change']].iplot(theme='white',filename='brennap_line_v1', world_readable=True,title='Line plot of changes in Global, Northern Hemisphere \nand Southern Hemisphere in Temperature',
                                     xTitle='Year',yTitle='Difference in degrees 0.01 Celsius When compared to average \nTemperature recorded between 1950 and 1980'
                                     )
##Year	Glob	NHem	SHem


# In[14]:

##lets try a spread chart
##kind='bar'
ddf2v0[['Global temperature change', 'Northern hemisphere temperature change', 'Southern hemisphere temperature change']].iplot(theme='white',filename='brennap_bar_line_v1', world_readable=True,title='Line plot of changes in Global, Northern Hemisphere \nand Southern Hemisphere in Temperature',
                                     kind='bar',xTitle='Year',yTitle='Difference in degrees 0.01 Celsius When compared to average \nTemperature recorded between 1950 and 1980'
                                     )


# In[15]:

import plotly.plotly as py
from plotly.graph_objs import *

Global = Scatter(
    x=xd.Year,
    y=xd.Glob
)
Northern_Hemisphere = Bar(
    x=xd.Year,
    y=xd.NHem
)
Southern_Hemisphere= Bar(
    x=xd.Year,
    y=xd.SHem
)
##Global,Northern_Hemisphere,Southern_Hemisphere


# In[16]:

data = Data([Global,Northern_Hemisphere,Southern_Hemisphere])
layout = Layout(
    
    barmode='stack',
    title='Temp bar',
    xaxis=XAxis(
        title='Year',
        titlefont=Font(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=YAxis(
        title='Temp Diff',
        titlefont=Font(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
fig=Figure(data=data,layout=layout)
plot_url = py.plot(fig, filename='brennap3-bar-linev2')


# In[17]:

plot_url


# In[18]:

from IPython.display import HTML
HTML('<iframe src=https://plot.ly/~brennap3/22 width=1400 height=900></iframe>')


# In[19]:

##does not look great lets go back to line chart


# In[20]:

from bs4 import BeautifulSoup
import urllib2
import re
import pandas as pd
##wiki=
wiki = "https://en.wikipedia.org/wiki/World_population_estimates"
header = {'User-Agent': 'Mozilla/5.0'} #Needed to prevent 403 error on Wikipedia
req = urllib2.Request(wiki,headers=header)
page = urllib2.urlopen(req)
soup = BeautifulSoup(page)


# In[21]:

tables = soup.find_all("table", { "class" : "wikitable" })


# In[22]:

import html5lib 


# In[23]:

table_1=tables[1]


# In[24]:

Year=[]
popRef=[]
UN=[]
Hyde=[]
Maddison=[]
Tanton=[]
Biraben=[]
McEvedy_johnson=[]
Thomlinson=[]
Durand=[]
Clark=[]
import requests


# In[25]:

for row in table_1.find_all('tr')[1:]:
    # Create a variable of all the <td> tag pairs in each <tr> tag pair,
    col = row.find_all('td')

    # Create a variable of the string inside 1st <td> tag pair,
    column_1 = col[0].find(text=True)
    # and append it to first_name variable
    Year.append(column_1)

    # Create a variable of the string inside 2nd <td> tag pair,
    column_2 = col[1].string
    # and append it to last_name variable
    popRef.append(column_2)

    # Create a variable of the string inside 3rd <td> tag pair,
    column_3 = col[2].string
    # and append it to age variable
    UN.append(column_3)

    # Create a variable of the string inside 4th <td> tag pair,
    column_4 = col[3].string
    # and append it to preTestScore variable
    Hyde.append(column_4)

    # Create a variable of the string inside 5th <td> tag pair,
    column_5 = col[4].string
    # and append it to postTestScore variable
    Maddison.append(column_5)
   
    # Create a variable of the string inside 5th <td> tag pair,
    column_6 = col[5].string
    # and append it to postTestScore variable
    Tanton.append(column_5)
   
    # Create a variable of the string inside 5th <td> tag pair,
    column_6 = col[6].string
    # and append it to postTestScore variable
    Biraben.append(column_6)
   
    # Create a variable of the string inside 5th <td> tag pair,
    column_7 = col[7].string
    # and append it to postTestScore variable
    McEvedy_johnson.append(column_7)
    
    # Create a variable of the string inside 5th <td> tag pair,
    column_8 = col[8].string
    # and append it to postTestScore variable
    Thomlinson.append(column_8)
    
    # Create a variable of the string inside 5th <td> tag pair,
    column_9 = col[9].string
    # and append it to postTestScore variable
    Durand.append(column_9)
    
    # Create a variable of the string inside 1st <td> tag pair,
    #column_10 = col[10].find(text=True)
    # and append it to first_name variable
    #Clark.append(column_10)
    #wont find get error
        


# In[26]:

Year_1=[]
US_1=[]
popRef_1=[]
UN_1=[]
Hyde_1=[]
Maddison_1=[]
Tanton_1=[]
Biraben_1=[]
McEvedy_jones_1=[]
Thomlinson_1=[]
Durand_1=[]
Clark_1=[]


# In[27]:

table_2=tables[2]


# In[28]:

for row in table_2.find_all('tr')[1:]:
    # Create a variable of all the <td> tag pairs in each <tr> tag pair,
    col = row.find_all('td')

    # Create a variable of the string inside 1st <td> tag pair,
    column_1 = col[0].find(text=True)
    # and append it to first_name variable
    Year_1.append(column_1)

    # Create a variable of the string inside 2nd <td> tag pair,
    column_2 = col[1].string
    # and append it to last_name variable
    US_1.append(column_2)

    # Create a variable of the string inside 3rd <td> tag pair,
    column_3 = col[2].string
    # and append it to age variable
    popRef_1.append(column_3)

    # Create a variable of the string inside 4th <td> tag pair,
    column_4 = col[3].string
    # and append it to preTestScore variable
    UN_1.append(column_4)

    # Create a variable of the string inside 5th <td> tag pair,
    column_5 = col[4].string
    # and append it to postTestScore variable
    Hyde_1.append(column_5)
   
    # Create a variable of the string inside 5th <td> tag pair,
    column_6 = col[5].string
    # and append it to postTestScore variable
    Maddison_1.append(column_5)
   
    # Create a variable of the string inside 5th <td> tag pair,
    column_6 = col[6].string
    # and append it to postTestScore variable
    Tanton_1.append(column_6)
   
    # Create a variable of the string inside 5th <td> tag pair,
    column_7 = col[7].string
    # and append it to postTestScore variable
    Biraben_1.append(column_7)
    
    # Create a variable of the string inside 5th <td> tag pair,
    column_8 = col[8].string
    # and append it to postTestScore variable
    McEvedy_jones_1.append(column_8)
    
    # Create a variable of the string inside 5th <td> tag pair,
    column_9 = col[9].string
    # and append it to postTestScore variable
    Thomlinson_1.append(column_9)
    
    # Create a variable of the string inside 5th <td> tag pair,
    column_10 = col[10].string
    # and append it to postTestScore variable
    Durand_1.append(column_10)
    
    column_11 = col[11].string
    # and append it to postTestScore variable
    Clark_1.append(column_11)
 


# In[29]:

columns_1 = {
            'Year_1':Year_1,
            'US_1':US_1,
            'popRef_1':popRef_1,
            'UN_1':UN_1,
            'Hyde_1':Hyde_1,
            'Maddison_1':Maddison_1,
            'Tanton_1':Tanton_1,
            'Biraben_1':Biraben_1,
            'McEvedy_jones_1':McEvedy_jones_1,
            'Thomlinson_1':Thomlinson_1,
            'Durand_1':Durand_1,
            'Clark_1':Clark_1
          }

# Create a dataframe from the columns variable
df = pd.DataFrame(columns_1)


# In[30]:

df['Year_1'] = df['Year_1'].str.replace(' ', '')


# In[31]:

df['UN_1'] = df['UN_1'].str.replace(' ', '')


# In[32]:

df['UN_1'] = df['UN_1'].str.replace(',', '')


# In[33]:

df['UN_1_NUM'] = df['UN_1'].convert_objects(convert_numeric=True)


# In[34]:

df['Year_1_NUM'] = df['Year_1'].convert_objects(convert_numeric=True)


# In[35]:

UN_POP=df[['UN_1_NUM','Year_1_NUM']]


# In[36]:

UN_POP.dtypes


# In[37]:

ddf2v0.dtypes


# In[38]:

##bind the following values
##1900 1,650,000,000
##1910 1,777,000,000
##1920 1,912,000,000
##1930 2,092,000,000
##1940 2,307,000,000
y = [{'UN_1_NUM':1650000000, 'Year_1_NUM':1900},
     {'UN_1_NUM':1777000000, 'Year_1_NUM':1910},
     {'UN_1_NUM':1912000000, 'Year_1_NUM':1920},
     {'UN_1_NUM':2092000000, 'Year_1_NUM':1930},
     {'UN_1_NUM':2307000000, 'Year_1_NUM':1940}]
UN_POP=UN_POP.append(y, ignore_index=True)


# In[39]:

UN_POP=UN_POP.sort(['Year_1_NUM'],ascending=[True])


# In[40]:

Global = Scatter(
    x=xd.Year,
    y=xd.Glob,
	name='Global Temperature Change'
)
Northern_Hemisphere = Scatter(
    x=xd.Year,
    y=xd.NHem,
	name='Northern Hemisphere Change'
)
Southern_Hemisphere = Scatter(
    x=xd.Year,
    y=xd.SHem,
	name='Southern Hemisphere Change'
)

Population_Stats = Scatter(
  x=UN_POP.Year_1_NUM,
  y=UN_POP.UN_1_NUM,
  name='UN Population statistics',
    mode='markers',           # show marker pts only
    marker=Marker(
    symbol='square' ,      # show square marker pts
    color='grey'
    ),
  yaxis='y2'
)

##Global,Northern_Hemisphere,Southern_Hemisphere



data = Data([Global,Northern_Hemisphere,Southern_Hemisphere,Population_Stats])

layout = Layout(
    title='Line plot of changes in Global, Northern Hemisphere \nand Southern Hemisphere in Temperature with \nUN global population statistics',
    yaxis=YAxis(
        title='Difference in degrees 0.01 Celsius \nwhen compared to average \ntemperature recorded between 1950 and 1980'
    ),
    yaxis2=YAxis(
        title='Global \npopulation \n(units \n1 billion)',
        titlefont=Font(
            color='rgb(148, 103, 189)'
        ),
        tickfont=Font(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    ),
	 xaxis=XAxis(
        title='Year',
        titlefont=Font(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
	)
	
)

fig=Figure(data=data,layout=layout)
##already ploted to plot uncomment below
plot_url = py.plot(fig, filename='brennap3-pop-temp-linev4')


# In[274]:

# Get this figure: fig = py.get_figure("https://plot.ly/~brennap3/80/")
# Get this figure's data: data = py.get_figure("https://plot.ly/~brennap3/80/").get_data()
# Add data to this figure: py.plot(Data([Scatter(x=[1, 2], y=[2, 3])]), filename ="brennap3-pop-temp-linev3", fileopt="extend"))
# Get y data of first trace: y1 = py.get_figure("https://plot.ly/~brennap3/80/").get_data()[0]["y"]

# Get figure documentation: https://plot.ly/python/get-requests/
# Add data documentation: https://plot.ly/python/file-options/

# You can reproduce this figure in Python with the following code!

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('username', 'api_key')
trace1 = Scatter(
    x=[1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014],
    y=[-22, -14, -17, -20, -28, -26, -25, -31, -20, -11, -34, -27, -31, -36, -32, -25, -17, -18, -30, -19, -13, -19, -29, -36, -43, -29, -26, -41, -42, -47, -45, -44, -40, -38, -22, -16, -36, -44, -31, -29, -27, -21, -29, -25, -24, -21, -8, -18, -16, -31, -11, -8, -11, -25, -9, -15, -10, 3, 5, 1, 6, 7, 5, 5, 13, 0, -8, -5, -11, -12, -19, -7, 1, 8, -12, -13, -18, 3, 5, 3, -4, 6, 4, 8, -19, -10, -4, -1, -5, 6, 4, -7, 2, 16, -7, -1, -12, 15, 6, 12, 23, 28, 9, 27, 12, 8, 15, 29, 36, 24, 39, 38, 19, 21, 29, 43, 33, 46, 62, 41, 41, 53, 62, 60, 52, 66, 60, 63, 49, 60, 67, 55, 58, 60, 68],
    name='Global Temperature Change',
    xsrc='brennap3:81:1328cb',
    ysrc='brennap3:81:1b29ec'
)
trace2 = Scatter(
    x=[1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014],
    y=[-33, -22, -23, -30, -42, -35, -33, -32, -21, -13, -37, -27, -37, -42, -35, -28, -21, -17, -28, -18, -8, -10, -30, -34, -42, -27, -21, -47, -42, -45, -44, -38, -48, -43, -20, -12, -36, -52, -35, -34, -27, -10, -26, -19, -13, -8, 5, -8, -4, -26, 7, 9, 3, -20, 5, -3, 2, 19, 23, 13, 14, 13, 11, 17, 24, 4, 1, 7, -1, -3, -17, 5, 6, 22, -4, -9, -25, 3, 15, 10, 7, 9, 17, 17, -19, -13, 0, 4, -4, -1, -3, -13, -17, 11, -19, -5, -22, 10, -1, 5, 13, 35, 3, 23, 2, -2, 11, 23, 34, 26, 48, 38, 10, 18, 35, 58, 27, 53, 72, 51, 50, 64, 71, 70, 65, 81, 77, 81, 63, 67, 84, 68, 74, 72, 86],
    name='Northern Hemisphere Change',
    xsrc='brennap3:81:1328cb',
    ysrc='brennap3:81:af3fe3'
)
trace3 = Scatter(
    x=[1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014],
    y=[-11, -6, -10, -10, -14, -16, -16, -30, -20, -10, -31, -27, -26, -30, -30, -23, -13, -18, -31, -20, -19, -28, -27, -39, -44, -30, -30, -36, -42, -49, -46, -49, -33, -32, -23, -20, -36, -36, -26, -24, -27, -31, -33, -32, -35, -34, -22, -28, -27, -36, -29, -25, -26, -30, -24, -26, -22, -13, -12, -12, -2, 2, -2, -7, 2, -4, -17, -17, -21, -20, -21, -18, -4, -6, -21, -16, -12, 2, -6, -5, -15, 2, -9, -2, -20, -8, -8, -6, -6, 13, 11, 0, 22, 21, 5, 3, -2, 20, 12, 19, 32, 21, 15, 31, 22, 19, 19, 34, 38, 23, 31, 38, 29, 25, 22, 29, 39, 38, 51, 31, 31, 42, 53, 50, 38, 51, 43, 44, 36, 52, 49, 43, 41, 48, 50],
    name='Southern Hemisphere Change',
    xsrc='brennap3:81:1328cb',
    ysrc='brennap3:81:2e3382'
)
trace4 = Scatter(
    x=[1900, 1910, 1920, 1930, 1940, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013],
    y=[1650000000.0, 1777000000.0, 1912000000.0, 2092000000.0, 2307000000.0, 2526000000.0, 2572850917.0, 2619292068.0, 2665865392.0, 2713172027.0, 2761650981.0, 2811572031.0, 2863042795.0, 2916030167.0, 2970395814.0, 3026002942.0, 3082830266.0, 3141071531.0, 3201178277.0, 3263738832.0, 3329122479.0, 3397475247.0, 3468521724.0, 3541674891.0, 3616108749.0, 3691172616.0, 3766754345.0, 3842873611.0, 3919182332.0, 3995304922.0, 4071020434.0, 4146135850.0, 4220816737.0, 4295664825.0, 4371527871.0, 4449048798.0, 4528234634.0, 4608962418.0, 4691559840.0, 4776392828.0, 4863601517.0, 4953376710.0, 5045315871.0, 5138214688.0, None, 5320816667.0, 5408908724.0, 5494899570.0, 5578865109.0, 5661086346.0, 5741822412.0, 5821016750.0, 5898688337.0, 5975303657.0, 6051478010.0, 6127700428.0, 6204147026.0, 6280853817.0, 6357991749.0, 6435705595.0, 6514094605.0, 6593227977.0, 6673105937.0, 6753649228.0, 6834721933.0, 6916183482.0, 6997998760.0, 7080072417.0, 7162119434.0],
    mode='markers',
    name='UN Population statistics',
    marker=Marker(
        color='grey',
        symbol='square'
    ),
    yaxis='y2',
    xsrc='brennap3:81:6773db',
    ysrc='brennap3:81:bc13bc'
)
data = Data([trace1, trace2, trace3, trace4])
layout = Layout(
    title='Line plot of changes in Global, Northern Hemisphere <br>and Southern Hemisphere in Temperature with <br>UN global population statistics',
    autosize=True,
    width=1032,
    height=657,
    xaxis=XAxis(
        title='Year',
        titlefont=Font(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        ),
        range=[1880, 2022.4749868351764],
        type='linear',
        autorange=True
    ),
    yaxis=YAxis(
        title='Difference in degrees 0.01 Celsius <br>when compared to average <br>temperature recorded between 1950 and 1980',
        range=[-59.66666666666667, 93.66666666666667],
        type='linear',
        autorange=True
    ),
    legend=Legend(
        x=1.0379213483146068,
        y=1.0524109014675052
    ),
    yaxis2=YAxis(
        title='Global <br>population <br>(units <br>1 billion)',
        titlefont=Font(
            color='rgb(148, 103, 189)'
        ),
        range=[1289320776.7226171, 7522798657.277383],
        type='linear',
        autorange=True,
        tickfont=Font(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
)
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
 
 
 
 
 
 


# My plot is here:
# https://plot.ly/~brennap3/108.embed
# The ipython notebook i used to create the visualization is here:
# https://gist.github.com/brennap3/f99ec3bb236e735cef2b
# Project Scope:
# Data visualization of global temperature deviations over time.
# Overview:
# Temperature Data from NASA (for global, Southern Hemisphere and Northern hemisphere) around temperature deviations (units deviations in 0.01 degree c compared to average temperature from 1950 to 1980) is plotted against time. There is also a second y axis which shows Global population (data scraped from https://en.wikipedia.org/wiki/World_population_estimates) statistics are also plotted over time. There appears to be a strong correlation with temperature deviation and global population figures.
# Types of data:
# The data used is both quantitative continuous (temperature deviation and global population) and quantitative discrete (year, the data was only collected between 1880 and 2013).
# 
# Visual encodings:
# Year : X axis positional
# Global population: Y axis positional
# Temperature deviation: Y axis (second y axis) positional
# 
# Type of plot used:
# A interactive scatter plot developed with python and plotly is used for the visualization. Plotly was chosen for a number of reasons:
# Interactivity could be introduced for very little overhead.
# Integrates easily with statistical programming languages like R or python.
# Abstracts D3.js libraries  allowing D3 based visualizations to be created easily. Plotly is easier to use and allows more functionality than comparative libraries like Bokeh, Vincent (Python) or ggvis (R), with the same level of use.
# D3 supports the use of SVG's (scalable vector graphics) which give a visually appealing presentation which renders well within a browsers.
# An interactive scatter plot was used as the data is best visually encoded using positional axis. Global and hemispherical temperature deviations are represented as continuous lines as a full data set between 1880 and 2013.  Global population are not represented as a continuous line but as just makers in a scatters plot, to emphasize that a full set of data from 1880 and 2013  for population statistics is not available. A color (grey) outside the color scheme for temperature is also used draw attention to the series  being different to the temperature data.
# 
# A title is used to explain the chart.
# All axis are labelled along with the units used.
# A legend is used to explain what the different series are.
# Color and style is used annotate the different series of data being used.
# 
# 
# Data Sources:
# data visualization coursera week 2 visualization
#  https://en.wikipedia.org/wiki/World_population_estimates

# In[ ]:



