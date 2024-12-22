    
# perform a function on one column among all columns
def partition(x):
    if x < 3:
        return 'positive'
    return 'negative'

column = df['column']
t = column.map(partition)
df["new_column"]=t

# another way:

def func(text):
    return text.lower()
df["reviews"] = df["reviews"].apply(func)

# <reviews> ki har value is function me jay gi or return aay gi, sab complete ho jay gi to output list of these returns df['reviews'] ko assign ho jay gi
--------------------------------------------------
# remove duplicates from specific columns

df = df.drop_duplicates(subset={"column1","column2","column3","column4"})
--------------------------------------------------
# is all values in particuler column is uqique or not?
df['column'].is_unique
# True / False
--------------------------------------------------
# df slicing
# in <column> from 1905 to 1906
df.loc[1905:, 'column'].head(10)
--------------------------------------------------
# column to numeric
df['column'] = pd.to_numeric(df['column'])
-------------------------------------------------
>>> pub = df['column']
>>> london = pub.str.contains('London')
>>> london[:5]
Identifier
206    True
216    True
218    True
472    True
480    True
Name: Place of Publication, dtype: bool

>>> oxford = pub.str.contains('Oxford')
df['column'] = np.where(london, 'London',
                                      np.where(oxford, 'Oxford',
                                               pub.str.replace('-', ' ')))

>>> df['column'].head()
Identifier
206    London
216    London
218    London
472    London
480    London
Name: Place of Publication, dtype: object
---------------------------------------------------
# rename one column name among all columns
df.rename(columns = {'column_x' : 'column_y'}, inplace=True)
# rename more than one columns names amont all columns
new_names = {'1' : 'column_1', '5' : 'column_5'}
df.reanem(column = new_names, inplace = True)
--------------------------------------------------
# Dropping rows having NULL Values
df.dropna(inplace=True)
--------------------------------------------------
encoding='cp1251'
encoding="ISO-8859-1"
df = pd.read_csv("file.csv", encoding = encoding, error_bad_lines=False)
--------------------------------------------------
# all columns and qty of null values in each column
df.isnull().sum()
--------------------------------------------------
# variables Qty that contain null values 
df.isnull().sum().sum()
--------------------------------------------------
# unique items Qty.
df.column.nunique()
-------------------------------------------------
# get grops sizes
df.groupby('column').size().reset_index(name='Enter new name for column')
# another 
df.groupby('column').count()
-------------------------------------------------
# entire column to lower case
df['column'] = df['column'].str.lower()
-------------------------------------------------
# string me jagan bhi <https> ho to us  k baad agly space tak sab remove kar do
import re
df[column] = df[column].apply(lambda i: re.sub(r"http\S+", "", i))
-------------------------------------------------
# all null values from all variables 
df.isnull().sum()
-------------------------------------------------
# can sort by values too
df.sort_values(by='B')
-------------------------------------------------
# column slicing
df.iloc[:,1:3]
-------------------------------------------------
# all rows and partuculer columns
# column slicing
df.iloc[:,1:3]
-------------------------------------------------
# filtering
df3 = df.copy()
df3['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
# srif wo rows jin k <E> column me <two> ya <four> h
df3[df3['E'].isin(['two', 'four'])]
-------------------------------------------------
# index me jahan bhi <amir> h us k <A> column me <0> kar do
df.at['amir','A'] = 0
-------------------------------------------------
# index me 'amir' or 'noman' k darmyan jo bhi rows hen un k column <E> ki value ko <1> kar do
df.loc['amir':'noman','E'] = 1
-------------------------------------------------
# drop rows with missing data
# agar ksi row ki koi value bhi <NA> ya <NaN> h to us poori row ko drop kar do
df.dropna(how='any')
-------------------------------------------------
# fill missing data
# sab missing values ko <5> me <5> fill kar do
df4.fillna(value=5)
-------------------------------------------------
# column vise mean
df.mean()
-------------------------------------------------
# pivot the mean calculation
# row vise mean # ignoring non numarical values
df.mean(1)
-------------------------------------------------
# <min> and <max> for all variables
df.apply(lambda x: (x.max(), x.min()))
-------------------------------------------------
# concatenation
pd.concat([df1, df2, df3, df4])
-------------------------------------------------
# group by
df.groupby('A').sum()
df.groupby('A').count()
-------------------------------------------------
scipy.sparse.csr_matrix to nd.array

type(count_train)
# scipy.sparse.csr.csr_matrix
new = count_train.A
type(new)
# numpy.ndarray
count_train.shape == new.shape
# True
-------------------------------------------------
# check 2 dataframes are equal or not
df1.equals(df2)
# True / False
-------------------------------------------------
# length of all rows in particular column
df['column'].str.len()
-------------------------------------------------
# length of all rows in particular column (for text column)
df['column'].str.len()
-------------------------------------------------
# count tokens in all rows in particular column (for text column)
df['column'].str.split().str.len()
-------------------------------------------------
# check particular word in each row in particular column
df['column'].str.contains('sub_string')
-------------------------------------------------
# check particular word occurence in each row in particular column
df['column'].str.count('word') # count <'word'> in each row in <'column'>
df['column'].str.count(r'\d') # count how many digits in each row in <'column'>
-------------------------------------------------
# har row k column <'column'> me dijits ki list
df['column'].str.findall(r'\d')
-------------------------------------------------
# we want to pull out the hour and minutes from each string:
df['column'].str.findall(r'(\d?\d):(\d\d)')
-------------------------------------------------
# har wo word jis k aakir me  <'day'> ho us ko <'???'> sy replace kar do
df['column'].str.replace(r'\w+day\b', '???')
-------------------------------------------------
# har wo word jis k aakhir me <'day'> aa raha ho us word k starting 3 characters k ela baqi charachters remove kar do
df['column'].str.replace(r'(\w+day\b)', lambda x: x.groups()[0][:3])
# first, we use the xpression that finds the weekday for us and we can create a group by placing it in parenthesis.
# then for the replacement parameter, we can pass in a lambda expression, we use goups to get a tuple of the groups, index 0 to get the first, only group, and slice the first three letters from the group. 
-------------------------------------------------
# extract time from every string, and create 2 new columns, one for hours, and second for minutes
df['column'].str.extract(r'(\d?\d):(\d\d)')
-------------------------------------------------
# extract time from every string, and create 3 new columns, one for hours,second for minutes and 3rd for am/pm, 
df['column'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))')
# since we use <extarctall> we exctact all times, even if 1 text contains more than 1 time
-------------------------------------------------
# extract time from every string, and create 3 new columns, one for hours,second for minutes and 3rd for am/pm,
# and set the names for new created columns 
df['column'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))')
# since we use <extarctall> we exctact all times, even if 1 text contains more than 1 time
------------------------------------------------- 
# new column with condition
df['new_column'] = np.where(df['old_column'] > 3, 1, 0)
# <old_column> jahan > 3, wahan <new_column> me <1> dal do,
else <0>
-------------------------------------------------
# pehly number sy pehly pehly tak ka sara string from dataframe column
import pandas as pd
from random import shuffle
t = list('hamzaamirsaleem1990')
lst = []
df = pd.DataFrame()
for i in range(10):
	shuffle(t)
	lst.append(''.join(t))
df['text'] = lst

def func(t):
	return t[:t.find(re.search('[0-9]+', t).group())]
    
df["text1"] = df["text"].apply(func)
-------------------------------------------------
# count plot group by sepecific column
# <df> k <column> ki grouping kar k un ka count ly lo os phir us counting ka <'bar'> graph bana do
df.column.value_counts().plot(kind='bar')
-------------------------------------------------
# pandas itrate over the rows
for i in df.iterrows():
    # now i is tuple, contains 2 values, first is index of this row, 
    # and second is a pandas Series, contain a content of this row.
    print(i, len(i), type(i), leni([0]), len(i[1]), type(i[0]), type(i[1]))
-------------------------------------------------
2 equal lenth lists to dataframe in single line
df = pd.DataFrame(data=list(zip(list1, list2)), columns=['column1', 'column2'])
-------------------------------------------------
DICT TO DATAFRAME
df
  index     Cost  Item Purchased  Name  Date  Delivered   Feedback
0   Store 1   22.5  Sponge    Chris   Jan   True    None
1   Store 1   2.5   Kitty Litter  Kevyn   Feb   True    Positive
2   Store 2   5.0   Spoon     Filip   March   True    Negative

a = pd.Series({0: 'December', 2:'mid.May'})
a

0    December
2     mid.May
dtype: object

adf['Date'] = a
adf

  index     Cost  Item Purchased  Name  Date  Delivered   Feedback
0   Store 1   22.5  Sponge    Chris   December  True  None
1   Store 1   2.5   Kitty Litter  Kevyn   NaN     True  Positive
2   Store 2   5.0   Spoon     Filip   mid.May   True  Negative
-------------------------------------------------
# DATAFRAME COLUMN RENAME
df = df.rename(columns={'REGION': 'region'})
-------------------------------------------------
# read tsv file using pandas
pd.read_csv('Location.tsv',delimiter='\t',encoding='utf-8')
-------------------------------------------------
# cpecify particuler column while loading csv file
df = pd.read_csv('file.csv', usecols = [5]) # getting only 5th column from csv file
-------------------------------------------------
# get only 5th column, and convert it to list
lst = pd.read_csv('file.csv', usecols = [5], encoding = "ISO-8859-1", error_bad_lines=False).iloc[:, 0].tolist()
-------------------------------------------------
# histogram of all numerical variables from data set
import matplotlib.pyplot as plt
train[train.dtypes[(train.dtypes=="float64")|(train.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])
-------------------------------------------------
# extract year from date column 
df['year'] = pd.DatetimeIndex(df['birth_date']).year
-------------------------------------------------
# columns Standardization 
scaler = StandardScaler() 
column_names_to_normalize = list(final_df.drop("DPD30\", axis=1).dtypes[(final_df.dtypes == 'int64') | 
                                       (final_df.dtypes == 'float64')].index) 
x = final_df[column_names_to_normalize].values 
x_scaled = scaler.fit_transform(x) 
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = final_df.index) 
final_df[column_names_to_normalize] = df_temp 
-------------------------------------------------
# new column based on existing column .................... data['Size_large'] = data.Size.map({'small':0, 'large':1})
-------------------------------------------------
# add new row
# ACCESS_METHOD_ID varibale me aksar values 2 dafa repeat hwi hen, magar kuch values 1 dafa i hen, jo 1 dafa hen un k lye 2nd line
# add karni h, jis me access_method_id call to same ho, or IsWeekDay me oposite ho(agar exist me ho to new me 1 and vise versa)
# or baqi columns me 0
def add_row(row):
    global df
    df.loc[-1] = row
    df.index = df.index + 1  
    
counts = df.ACCESS_METHOD_ID_.value_counts()
for i in counts[counts == 1].index:
    m = 0 if df[df.ACCESS_METHOD_ID_ == i].IsWeekDay_.values == 1 else 1
    add_row([i, m, 0, 0, 0, 0, 0, 0, 0,  0])
-------------------------------------------------
count and proportion of values in a column:
import researchpy as rp
rp.summary_cat(df)
-------------------------------------------------
Add a prefix to all of your column names:
df.add_prefix('X_')
Add a suffix to all of your column names:
df.add_suffix('_Y')
-------------------------------------------------
# continues variables to catagorical with bins...
df['age_groups'] = pd.cut(df.age, bins=[0, 18, 65, 99], labels=['child', 'adult', 'elderly'])
-------------------------------------------------
df = df.replace(r'^\s*$', " ", regex=True)
-------------------------------------------------
# fill diffirent  variables with distinct value
df1.fillna({"column_x":0.5,
	"column_n":0})
-------------------------------------------------
# select only <object> type variables:
ds_cat = dataset.select_dtypes(include = 'object').copy()
-------------------------------------------------
df.ix[::2,0] = np.nan # in column 0,     set elements with indices 0,2,4, ... to NaN
df.ix[::4,1] = pd.NaT # in column 1,     set elements with indices 0,4, ... to np.NaT
df.ix[:3,2] = 'nan'   # in column 2,     set elements with index from 0 to 3 to 'nan'
df.ix[:,5] = None     # in column 5,     set all elements to None
df.ix[5,:] = None     # in column row 5, set all elements to None
df.ix[7,:] = np.nan   # in column row 7, set all elements to NaN


# select vars based on dtypes ................. features.select_dtypes('number')
----------------------------------------------------------------
# join two dataframes using <join>
df1 = pd.DataFrame({"A" : list(range(10)), "B" : list('abcdefghij')})
df2 = pd.DataFrame({"A" : list(range(10, 20)), "B" : list('klmnopqrst')})

full_df = df.join(df1, lsuffix='_caller', rsuffix='_other')
----------------------------------------------------------------

# How do I subtract the previous row from the current row in a pandas dataframe and apply it to every row; without using a loop? ........................... df["variable"].diff()


# display all rows .................. from IPython.display import display ; pd.set_option('display.max_rows', 10000); display(df)

# Convert a column of datetimes to epoch .................. (df['date'] - dt.datetime(1970,1,1)).dt.total_seconds()
# append in existing csv file .............. df.to_csv(existing_file_name.csv, mode='a', header=False)

# drop duplicates excluding column .................... https://stackoverflow.com/questions/38445416/dropping-duplicates-in-pandas-excluding-one-column .................. df.drop_duplicates(subset=['City', 'State', 'Zip', 'Date'])  ............................. df.drop_duplicates(subset=df.columns.difference(['Description']))

# line plot for all variables in dataframe vs one variable in x-axis ................ df.set_index('x_axis_column').plot(figsize=(17,10), grid=True)
# week number (from date columns) .............. df['Date'].dt.week
# month (from date columns) .............. df['Date'].dt.month
# year  (from date columns) .............. df['Date'].dt.year# convert daily data to weekly ................ df.resample('M', on='week_starting').mean() # week_starting is column name 
----------------------------------------------------
# plot multiple plots in one graph
plt.figure(figsize=(15, 7))  # make separate figure
plt.subplot(2, 1, 1)
plt.plot(dF["3"])
plt.xlabel('t')
plt.ylabel('y')
plt.title('Top figure')

plt.subplot(2, 1, 2)
plt.plot(dF["MAPE"])
plt.xlabel('t')
plt.ylabel('y')
plt.show()

---------------------------------------------------
# standrize all numeric variables in df ............. df[df.select_dtypes("number").columns] = df.select_dtypes("number").apply(lambda x:(x - x.mean()) / x.std())
# jab ham matplotlib sy plotting kar rahy hoty hen to kafi unnecessory sheezen print ho rahi hoti hen; like: <matplotlib.lines.Line2D at 0xXXXXXXXX>. is ko remove karny k lye plot ki command k end me semi-colon laga den; eg: df.select_dtypes("number").plot();
# drop columns range by start and end names ......... df.drop(df.loc[:, "2019":"Unnamed: 61"].columns, axis = 1) 
# display at max 999 rows ...... pd.options.display.max_rows = 999
# drop duplicated columns ............. df = df.loc[:,~df.columns.duplicated()]

# todays date ........ pd.to_datetime("today")

# total number of cells in dataframe: df.size
# null/nan/none values in datetime column ............. np.isnat(df.date_column_name)
#------------
# merge multiple dataframes in one
from functools import reduce
dfs = [df1, df2, df3, ...]
# solution 1
result_1 = pd.concat(dfs, join='outer', axis=1)
# solution 2
result_2 = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, left_index=True, right_index=True, how='outer'), dfs)
#------------

-------------------------------------------------
# https://www.kdnuggets.com/2020/09/introduction-time-series-analysis-python.html#.X2zoteVcRVk.linkedin

# Indexing in Time-Series Data
df.loc['2000-01-01':'2015-01-01']

# Let’s say we want to get all the data of all the first months from 1992-01-01 to 2000-01-01. The syntax for this in Pandas is ['starting date':'ending date':step].
df.loc['1992-01-01':'2000-01-01':12]

# Time-Resampling using Pandas
# Think of resampling as groupby() where we group by based on any column and then apply an aggregate function to check our results. Whereas in the Time-Series index, we can resample based on any rule in which we specify whether we want to resample based on “Years” or “Months” or “Days or anything else.
# Some important rules for which we resample our time series index are:
  # M = Month End
  # A = Year-End
  # MS = Month Start
  # AS = Year Start
# Let’s say we want to calculate the mean value of shipment at the start of every year. We can do this by calling resample at rule='AS' for Year Start and then calling the aggregate function mean on it.
# We can see the head of it as follows.
df.resample(rule='AS').mean().head()
# now we have the mean of Shipping at the start of every year.

# We can even use our own custom functions with resample. Let’s say we want to calculate the sum of every year with a custom function. We can do that as follows.
def sum_of_year(year_val):
    return year_val.sum()
# And then we can apply it via resampling as follows.
df.resample(rule='AS').apply(year_val)

# Rolling Time Series
# Rolling is also similar to Time Resampling, but in Rolling, we take a window of any size and perform any function on it. In simple words, we can say that a rolling window of size k means k consecutive values.
# Let’s see an example. If we want to calculate the rolling average of 10 days, we can do it as follows.
df.rolling(window=10).mean().head(20) # head to see first 20 values 
# Now here, we can see that the first 10 values are NaN because there are not enough values to calculate the rolling mean for the first 10 values. It starts calculating the mean from the 11th value and goes on.
-------------------------------------------------

# select columns with regex ......... df.filter(regex='^(CURRENT_STATUS)', axis=1)
# sorting-by-absolute-value-without-changing-the-data ............. df.reindex(df.b.abs().sort_values().index) .................. https://stackoverflow.com/questions/30486263/sorting-by-absolute-value-without-changing-the-data

# join multiple columns in one column .......... df.agg(" ".join, axis=1)
# replace multiple values in string at once with regex ................ df.Column.replace({"\s?No Country\s?|\s?No Brand\s?": ""}, regex=True)
# replace multiple values in string at once ......... df.column.str.replace(r'No Brand\s*|\s*No Country', '')

# value_counts has perameter <normalize> when this set to True the output in percentages rather then actual counts

# get todays date and time : pd.to_datetime("now") ..... today = pd.to_datetime("today")

# search multiple substrings in string cell .......... df.OBJ_column.str.contains("sub_string_1|substring_2|substring_3")
# draw line in plot ....... plt.axvline(x=0.22058956) ............ plt.axhline(y=df.col.mean());

#---------------------
# Looking up the list of sheets in an excel file .........
xl = pd.ExcelFile('foo.xls')
xl.sheet_names  # see all sheet names
xl.parse(sheet_name)  # read a specific sheet to DataFrame
# >>>>>>>>>>>>>>>>>>>>>



# round float (5 floats) .............. pd.set_option('display.float_format', lambda x: '%.3f' % x)


#-------------------
# read/write datafram from/to gzip compressed file
df.to_csv(path_or_buf = 'sample.csv.gz', compression="gzip", index = None)
pd.read_csv("sample.csv.gz", compression="gzip")
#---------END
>> df.createdtimestamp.head(2)
0        2020-03-10 22:25:48
1        2020-03-10 22:25:48

>> df.createdtimestamp.head(2).str.split().str[0]
0        2020-03-10
1        2020-03-10
#------------------------
#droping missing values rows/columns
threshold = 0.7 #Dropping columns with missing value rate higher than threshold
data = data[data.columns[data.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
data = data.loc[data.isnull().mean(axis=1) < threshold]
#-----------------------
#Filling missing values with medians of the columns
data = data.fillna(data.median())
#----------------------
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim = data['column'].mean () + data['column'].std () * factor
lower_lim = data['column'].mean () - data['column'].std () * factor

data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]

# In addition, z-score can be used instead of the formula above. 
# Z-score (or standard score) standardizes the distance between a value and the mean using the standard deviation.
#---------------------
#Dropping the outlier rows with Percentiles
upper_lim = data['column'].quantile(.95)
lower_lim = data['column'].quantile(.05)

data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]
#---------------------
#Capping the outlier rows with Percentiles
upper_lim = data['column'].quantile(.95)
lower_lim = data['column'].quantile(.05)data.loc[(df[column] > upper_lim),column] = upper_lim
data.loc[(df[column] < lower_lim),column] = lower_lim
#---------------------
#Numerical Binning Example
data['bin'] = pd.cut(data['value'], bins=[0,30,70,100], labels=["Low", "Mid", "High"])
#---------------------
#Categorical Binning Exampl
e     Country
0      Spain
1      Chile
2  Australia
3      Italy
4     Brazil
conditions = [
    data['Country'].str.contains('Spain'),
    data['Country'].str.contains('Italy'),
    data['Country'].str.contains('Chile'),
    data['Country'].str.contains('Brazil')]

choices = ['Europe', 'Europe', 'South America', 'South America']

data['Continent'] = np.select(conditions, choices, default='Other')
       Country      Continent
0      Spain           Europe
1      Chile    South America
2  Australia            Other
3      Italy           Europe
4     Brazil    South America
#--------------------
# drop rows contains outlier
# using z score
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(boston_df))
df_without_outliers = df[(z < 3).all(axis=1)]

# using IQR
df = df[~(
	(df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
	).any(axis=1)]

#-------------------
# two numerical variables plot on top of each other (stack)
df1[['x', 'y']].plot.hist(alpha=0.3, stacked=True, bins=10)
#-------------------
# kmean clustring
model = KMeans(n_clusters=3, max_iter=300)
model.fit(df)

# Plot the First iteration of the kmean.
colormap = np.array(['red', 'lime', 'blue'])
plt.scatter(df.x, df.y, c=colormap[model.labels_], s=20)


# extract date columns.
df = df.assign(hour  = df['date'].dt.hour,
               day   = df['date'].dt.day,
               month = df['date'].dt.month,
               year  = df['date'].dt.year)
  

# jab ham size reduce karny k lye str/object type ko catagory me convert karty hen, to phir us ko <df.select_dtypes("O")> ya <df.select_dtypes(str)> sy access nahi kar sakty, rather we should use <df.select_dtypes(pd.CategoricalDtype)>
# remove all non numeric characters from column ........................ df.column.str.replace(r'\D+', '')
# convert all columns to string ............. X = X.astype(str)

# groupby and count/size
df.groupby("Province").size() # return a series
df.groupby("Province").count() OR df.groupby("Province").agg("count") # return a data frame, includeing all original columns except those grouped by


df.reset_index(name='new_column_name') # df.reset_index() sy existing index dataframe me add ho jata h (with name <index>, agar ham is ko koi name dena chahye hen to <name> k argument sy is ko kar sakty hen)

df.groupby("col_1").agg({ 'col3': ['mean', 'count'], 'col4': ['median', 'min', 'count']})

s = pd.Series(["a", "b", "c"]).to_frame(name='counts') # <s> is data frame with single column <counts>
# zaroori nahi k dask hamesha pandas sy faster ho, is image <dask-vs-pandas.png> ko check karen.

# more then one missing value (NA) petters while readeing a file ................. missing_value_formats = ["n.a.","?","NA","n/a", "na", "--"]; df = pd.read_csv("employees.csv", na_values = missing_value_formats)




pd.Series(['a', 'm', 'i', 'r', ' ', 's', 'a', 'l', 'i', 'm']).sum() ........... convert text series to one string

# replace NaNs by previous values .............. df.fillna(method='ffill') ........... df.column_name.fillna(method='ffill') 


# remove duplicated columns ............ df = df.loc[:,~df.columns.duplicated()]


# read 1% data with random rows from big csv-file ............... df = pd.read_csv('file.csv', skiprows = lambda x: x>0 and np.random.rand() > 0.01)


# Internally process the file in chunks (low_memory) .........  Internally process the file in chunks, resulting in lower memory use while parsing, but possibly mixed type inference. To ensure no mixed types specify the type with the dtype parameter. ................ df = pd.read_csv('file.csv', low_memory=True, dtype = {'text': str,  'env_problems': 'Int64', 'pollution': 'Int64',  'treatment': 'Int64', 'climate': 'Int64', 'biomonitoring': 'Int64'})


# select 0,1,4,7th columns .............. df.iloc[:, np.r_[0:3, 4:7]]


# give perameter with apply : df.application_date.agg([min,max]).apply(pd.to_datetime, unit='s')

#----------------------
>>> df
        one  two  three
mouse     1    2      3
rabbit    4    5      6

>>> # select columns by name
>>> df.filter(items=['one', 'three'])
         one  three
mouse     1      3
rabbit    4      6

>>> # select columns by regular expression
>>> df.filter(regex='e$', axis=1)
         one  three
mouse     1      3
rabbit    4      6

>>> # select rows containing 'bbi'
>>> df.filter(like='bbi', axis=0)
          one  two  three
rabbit    4    5      6
#---------------------


# search multiple substrings (startsiwth) in string cell ............. prefixes = ["xyz", "abc"]; "abcde".startswith(tuple(prefixes))

# convert float to int with NaNs ....................... df.column.astype(float).astype('Int64') ............. df.column.astype(float).astype(int, errors='ignore')

pd.set_option('max_rows', 5)
x.replace({[0,4]:'AAA'})
x.replace({range(40,99):'AAA'})

reviews_written = reviews.groupby('taster_twitter_handle').size()
or
reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

Set the index name in the dataset to 'A_NEW_INDEX_NAME'.
df.rename_axis("A_NEW_INDEX_NAME", axis='rows')

# print dataframe neatly ......... print(df.to_markdown(tablefmt="grid"))
# is_unique consider NaNs, so if all values are unique but there are one or more NaNs, pd.Series.is_unique returns False, and pd.Series.dropna().is_unique returns True

# infinite values to None  .............  dfdf.replace([np.inf, -np.inf], np.nan)
# Split (expend/extend) a  column of lists into multiple columns ............. df[['col1','col2']] = pd.DataFrame(df.col.tolist(), index= df.index)

# select the salary of the employee with id number 478 by position .......... df_employees.iat[1, 3]
# select the salary of the employee with id number 478 by label ............. df_employees.at['478', 'salary']
# remove all non-numeric characters from all the values in a particular column  ..................... dfObject['C'] = dfObject['C'].str.replace(r'\D+', '') # Or, since in Python 3, \D is fully Unicode-aware by default and thus does not match non-ASCII digits (like ۱۲۳۴۵۶۷۸۹, see proof) you should consider ................. dfObject['C'] = dfObject['C'].str.replace(r'[^0-9]+', '')
# Drop columns whose name contains a specific string ................ df.filter(regex='^((?!badword).)*$') .............. df[df.columns.drop(list(df.filter(regex='Test', case=False)))] ............. df.select(lambda x: not re.search('Test\d+', x), axis=1)

# remove all alphanumeric from series ............ df.col.str.replace('[a-zA-Z0-9]', '')
# head and tail at the same time ............. df.iloc[np.r_[0:5, -5:0]]

# remove all non alpha from series ................ df.column_namestr.replace(r'[^a-zA-Z]', '')


# change background color to black ........... plt.style.use('dark_background')

####### show image in full screen 
plt.switch_backend('TkAgg'); df.my_col.plot(color='red');
mng = plt.get_current_fig_manager()
### works on Ubuntu??? >> did NOT working on windows
mng.resize(*mng.window.maxsize())
# mng.window.state('zoomed') #works fine on Windows!
plt.show()
###################################


# month name ........... df.date_col.dt.month_name()
# isocalendar ............ df.date_col.dt.isocalendar()

# merge multiple dataframes at once ............. import functools as ft; df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='name'), dfs)


# get hash of columns ............ hash(tuple(my_df.my_col))

# COLOR : ...................... from termcolor import colored; print(colored('this will print in green', 'green'))

# download a page (only text) ........... os.system(f"curl -s {last_episode_link} > /home/amir/ab__.txT"

# get hash of a given file ............. hashlib.sha224(open(file_name, 'rb').read()).hexdigest()


# read the last byte ............ open(file_name, 'rb').seek(-1, os.SEEK_END)

# size of a file ................. os.path.getsize(val)

# memory | swap | ram ........... psutil.virtual_memory()

# except KeyboardInterrupt



#check  is there any duplicated column ............... df.apply(lambda col: hash(tuple(col))).is_unique


# exclude columns with regex ............ regex ='^((?!_1_|EMA_2_close_).)*$'; df.filter(regex=regex) # remove columns that have 'EMA_2_close_' or '_1_' in their names. 
# pandas does sort series by index before ploting internaly, to avoid this use ....... ts.sort_values().plot(use_index=False)


# Highlighting rows based on a condition
def print_with_highlighted_rows(self, condition):
    highlighted_rows = np.where(condition,'background-color: green','')
    styler = self.df.style.apply(lambda _: highlighted_rows)
    styler = styler.set_properties(**{'text-align': 'left'})
    display(styler)


# pandas data frame view in external window like R-view : pandasgui ........... pip3 install pandasgui ........... from pandasgui import show ....... show(my_df)


# Setting Yaxis in Matplotlib using Pandas ................... df.plot(ylim=(0,200))


# Adding a y-axis label to secondary y-axis in matplotlib ............... plot(y1) ; plt.gca().twinx().plot(y2, color = 'r') # default color is same as first ax

# find missing dates from column/index ........... pd.date_range(X_train.index.min(), X_train.index.max(), freq='D').difference(X_train.index)

# get date component from datetime column .............. df[['year', 'month', 'day', 'hour', 'minute', 'sec']] = pd.to_datetime(df['time']).apply(lambda x: (x.year, x.month, x.day, x.hour, x.minute, x.second)).apply(pd.Series)

# df['column'].str.contains('abc', case=False)

# loc with multiindex ................ df.loc[('index_1', "index_2")]  ............. df.loc[ [('at', 3),('at', 7),('at', 5)] ]

# fill missing values with mean: .......... df.fillna(df.select_dtypes(include='number').mean().iloc[0], inplace=True)
# fill msising values with mode ............. df.fillna(df.select_dtypes(include='object').mode().iloc[0], inplace=True)


# pd.read_csv reads .zip file if there is only one csv file, likewise pd.read_pickle reads .zip file if there is only one pkl file


# data_df.col.replace({1:'Survived'}, inplace=True)


# import pandas as pd

# from io import StringIO; data="Amir,32\nHamza,8\nOqba,4\nUrwa,2"; pd.read_csv(StringIO(data), names=["name", "age"])
#     name  age
# 0   Amir   32
# 1  Hamza    8
# 2   Oqba    4
# 3   Urwa    2



# sort dataframe by multiple columns, some ascending and some descending .......... df.sort_values(by=["col_a", "col_2", "col_3"], ascending=[True, False, False])

# horizontal box plots in matplotlib ............  df.col.boxplot(vert=False) ........... plt.boxplot(df["feature"], vert=False)

## ################# mask
is_small = col < col.quantile(.25)
is_large = col > col.quantile(.75)
is_medium = ~(is_small | is_large)
col.mask(is_small, 'small').mask(is_large, 'large').mask(is_medium, 'medium')


is_small = df < df.quantile(.25)
is_large = df > df.quantile(.75)
pd.DataFrame(
    np.where(is_small, 'small', np.where(is_large, 'large', 'medium')),
    df.index, df.columns
)


(pd.qcut(col, q=[0, .25, .75, 1], labels=['small', 'medium', 'large']))

df.apply(pd.qcut, q=[0, .25, .75, 1], labels=['small', 'medium', 'large'], axis=0)
########################


# convert float to str with preserving NaNs ............ df.my_float_col = df.my_float_col.dropna().astype(int).astype(str).reindex(df.index)


# manual grid search dataframe from dictionary ................... from itertools import product; param_combinations = list(product(*hyperparameters.values())); pd.DataFrame(param_combinations, columns=hyperparameters.keys())


# Filter for the year 2019 with loc
df.loc[lambda x: x.Year == 2019]




>>> x
Quality        
bad      Size      1.00
         Weight   -0.01
good     Size      1.00
         Weight   -0.28
Name: Size, dtype: float64

>>> x.xs("Weight", level=1)
Quality
bad    -0.01
good   -0.28
Name: Size, dtype: float64

>>> x.xs("bad", level=0)
Size      1.00
Weight   -0.01
Name: Size, dtype: float64



df[df['School Name'].str.contains('Academy', case=False)]


pd.Series([1,2,33,983,43,9]).astype(str).str.zfill(10)


# Identify the top-rated applications by total number of ratings for each primary genre.
df.loc[df.groupby("prime_genre")["rating_count_tot"].idxmax()]



# Sort the DataFrame based on the absolute value of coefficients
coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)


df.groupby([df.date.dt.date, 'category']).price.mean()



filtered_df = filtered_df.sort_values(by = ['life_expec', 'health'], ascending = [True, False])



# Create bins for child mortality, ensuring the last bin includes the maximum value
bins = np.arange(0, df['child_mort'].max() + 10, 10)
# Create a new column with the binned child mortality values
df['child_mort_bins'] = pd.cut(df['child_mort'], bins=bins, right=False, labels=bins[:-1])



df_filtered['baseline_value_boxcox'], _ = boxcox(df_filtered['baseline value'])






filtered_df = filtered_df.sort_values(by='z_score', key=abs, ascending=False)



df_agg = df_real_estate.groupby('TRANSACTION TYPE').agg(
    Number_of_Transactions = ('TRANSACTION TYPE', 'count'),
    Average_Transaction_Amount_INR = ('AMOUNT IN (INR)', 'mean'),
    Average_Client_Age = ('CLIENT AGE', 'mean')
).reset_index()



df.groupby('Segment')[['Discount', 'Profit']].describe()


sns.boxplot(x='Segment', y='Discount', data=data)




df.apply(pd.to_numeric, errors='coerce')



# Combine Booker, Matias, and Nicolas into a single column Seller, taking the first non-empty value for each row.
df['Seller'] = df[['Booker', 'Matias', 'Nicolas']].apply(lambda x: next((i for i in x if i), ''), axis=1)



df['FOUR_DAY_TOTAL'] = df['ROUNDED_AMOUNT'].rolling('4D', closed='left').sum()


end_date = start_date + pd.DateOffset(days=3)


# Remove '$', and 'nan'
df.column_name.astype(str).str.replace(r'[$, nan]', '', regex=True)





# Find the most common `Type` for each `Listing ID`
(
    (
        df.groupby(['Listing ID', 'Type'])
        .size()
        .reset_index(name='Count')
        .sort_values(by='Count', ascending=False)
    )
    .groupby('Listing ID')
    .first()
    .reset_index()
    [['Listing ID', 'Type']]
)






-------
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

categorical_features = ['race', 'sex', 'marital.status']
numeric_features = ['age', 'hours.per.week']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create the logistic regression pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


feature_names = (numeric_features +
                 list(pipeline.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names(input_features=categorical_features)))

# Get coefficients from the logistic regression model
coefficients = pipeline.named_steps['classifier'].coef_[0]

# Create a DataFrame to view the feature names with their corresponding coefficients
coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
-------


>>> from scipy.stats import kendalltau
>>> kendalltau?
mannwhitneyu(
    x,
    y,
    use_continuity=True,
    alternative='two-sided',
    axis=0,
    method='auto',
    *,
    nan_policy='propagate',
    keepdims=False,
)
Docstring:
Perform the Mann-Whitney U rank test on two independent samples.

The Mann-Whitney U test is a nonparametric test of the null hypothesis
that the distribution underlying sample `x` is the same as the
distribution underlying sample `y`. It is often used as a test of
difference in location between distributions.



# Kind of label encodeing
pd.factorize(df['occupation'])[0]
pd.Categorical(df['income']).codes




>>> from scipy.stats import kendalltau
>>> kendalltau?
mannwhitneyu(
    x,
    y,
    use_continuity=True,
    alternative='two-sided',
    axis=0,
    method='auto',
    *,
    nan_policy='propagate',
    keepdims=False,
)
Docstring:
Perform the Mann-Whitney U rank test on two independent samples.

The Mann-Whitney U test is a nonparametric test of the null hypothesis
that the distribution underlying sample `x` is the same as the
distribution underlying sample `y`. It is often used as a test of
difference in location between distributions.



# Kind of label encodeing
pd.factorize(df['occupation'])[0]
pd.Categorical(df['income']).codes





reset_index(name='robbery_count')




for i, (year, counts) in enumerate(robbery_counts_by_year.items()):




precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')



from IPython.display import HTML
import base64

# Replace 'path/to/your/image.jpg' with the actual path to your image file
image_path = '/home/amir/26.png'

# Read the image file and convert it to base64
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Embed the base64-encoded image in the notebook
html_code = f'<img src="data:image/jpeg;base64,{encoded_image}">'
# HTML(html_code)
html_code



from scipy.stats import zscore
df['avg_rating_zscore'] = zscore(df['Book_average_rating'])






import pandas as pd
from scipy import stats

df = pd.read_csv('/tmp/inputs/1_raw_base_data.csv')

z_avg_rating = stats.zscore(df['Book_average_rating']) 
z_gross_sales = stats.zscore(df['gross sales'])

threshold = 3
filtered_df = df[(z_avg_rating < threshold) &  
                (z_avg_rating > -threshold) &  
                (z_gross_sales < threshold) &
                (z_gross_sales > -threshold)]

print(filtered_df.shape)





plt.scatter(df['units sold'], df['Book_average_rating'])




df['Sales_Category'] = pd.qcut(df['gross sales'], q=[0, 0.33, 0.66, 1], labels=['Low', 'Medium', 'High'])


df._get_numeric_data()



df = df[np.isfinite(df['gross sales'])]



df = df.replace([np.inf, -np.inf], np.nan).fillna(0)   



df['Suspension_100%_Indicator'] = df['Description'].str.contains('Suspension 100%', case=False)




data = pd.get_dummies(serious_accidents, columns=['Road_Surface_Conditions'], drop_first=True)






time_intervals = ['Morning', 'Afternoon', 'Evening', 'Night']

time_counts = serious_accidents.groupby(pd.cut(pd.to_datetime(serious_accidents['Time'], format='%H:%M').dt.hour,
                                                bins=[0, 6, 12, 18, 24], labels=time_intervals, include_lowest=True)).size()




# Regression plot
sns.regplot(x='Age at enrollment', y='Curricular units 1st sem (grade)', data=data, ax=axes[i])






skew = df['PRICE'].skew()  
# Assess normality
kstest = stats.kstest(df['PRICE'], 'norm')
print(f'Normality test p-value: {kstest.pvalue:.3f}')






>>> d1 = df.groupby(['ADDRESS', 'BROKERTITLE'], as_index=False).first()
>>> d2 = df.drop_duplicates(subset=['ADDRESS', 'BROKERTITLE'], keep='first')
>>> d1 = d1.sort_values(['ADDRESS', "BROKERTITLE"]).reset_index(drop=True)
>>> d2 = d2.sort_values(['ADDRESS', "BROKERTITLE"]).reset_index(drop=True)
>>> d1.eq(d2).all().all()
# True






series.reset_index(name='counts')
df.reset_index(names=['counts'])






X._get_numeric_data()   


df['Gender'] = pd.factorize(df['Gender'])[0]


df[['Year', 'Quarter', 'Month', 'Day', 'Weekday']] = pd.to_datetime(df['Date']).apply(lambda x: pd.Series([x.year, x.quarter, x.month, x.day, x.weekday()]))









df['Quarter'] = df['Date'].dt.to_period("Q")


df.hist(bins=20, figsize=(10, 8)) # all numeric variables.


sns.scatterplot(data=df, x="total", y="tax", hue="service")









from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer(output_distribution='normal')
df[num_cols] = scaler.fit_transform(df[num_cols])


from sklearn.preprocessing import KBinsDiscretizer
kb_discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal')
df[num_cols] = kb_discretizer.fit_transform(df[num_cols])




df.columns.difference(['InvoiceNo'])



> medians
{'Tuesday': 9.32,
 'Monday': 8.4,
 'Thursday': 10.5,
 'Sunday': 6.35,
 'Wednesday': 9.92,
 'Friday': 9.92}
> max_median_day = max(medians, key=medians.get)
> max_median_day
Thursday
> medians[max_median_day]
10.5



df.groupby('CustomerID').agg(No_of_invoices=('InvoiceNo', 'nunique'), Total_Quantity=('Quantity', 'sum')).reset_index()





df[numeric_cols] = df[numeric_cols].replace({np.inf: np.nan, -np.inf: np.nan})


gift_products = df[df['Description'].str.contains('GIFT', case=False, na=False)]









# rating: string
# runtime: numeric
fig, ax = plt.subplots(figsize=(10,6))
for rating, data in df.groupby('rating'):
    data['runtime'].plot(kind='kde', label=rating, ax=ax)
ax.set_xlabel('Runtime (minutes)')  
ax.set_ylabel('Density')
ax.set_title('Distribution of Movie Runtimes by Rating')
ax.legend()
plt.tight_layout()
plt.show()


_, ax = plt.subplots(figsize=(8,6))
df['runtime'].hist(by=df['rating'], ax=ax, bins=30)
ax.set_xlabel('Runtime (minutes)')
ax.set_title('Distribution of Movie Runtimes by Rating')
plt.tight_layout()
plt.show()



category_engagement = df.groupby('category_id').agg(
    videos=('video_id', 'count'),
    avg_views=('views', 'mean'), 
    avg_likes=('likes', 'mean'),
    avg_dislikes=('dislikes', 'mean'),
    avg_comments=('comment_count', 'mean')
).reset_index()



















def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


word_counts = Counter()
for desc in df['description']:
    word_counts.update(desc.split())

sns.heatmap(train.isnull())





# Create a dataframe `df_top20` for the top 20 rows with highest value of `Total`
df_top20 = df.nlargest(20, "Total")


df = df.asfreq('b','ffill')
kurt=df['Close'].kurtosis()



corr = df.corr()
corr.style.background_gradient(cmap = "copper")




sns.heatmap(df.corr(), annot = True, square = False, linewidths = 5, linecolor = "white", cmap = "Oranges");

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = (14,8)
sns.set_style('darkgrid')




fig,axes = plt.subplots(2,2,figsize=[15,7])

axes[0,0].plot(data.Close)
axes[0,0].set_title("Daily",size=16)
axes[0,1].plot(data.Close.resample('M').mean())
axes[0,1].set_title("Monthly",size=16)


axes[1,0].plot(data.Close.resample('Q').mean())
axes[1,0].set_title('Quarterly',size=16)

axes[1,1].plot(data.Close.resample('A').mean())
axes[1,1].set_title('Annualy',size=16)

plt.tight_layout()
plt.show()







data[['a', 'b']] = data['c'].str.split('/', expand=True)



# Extract components from `Date`
date_components = ["Year", "Month", "Day"]
for component in date_components:
  df[component] = getattr(df['Date'].dt, component.lower())



# Extract the month.
df['month'] = df['date'].dt.strftime('%B')



merged_df.groupby('A').agg(
    {'B':'mean',
     'C':'mean'
    }
)



# Get percentile
from scipy.stats import percentileofscore
desired_value = 7
series = pd.Series(range(20))
percentile_rank = percentileofscore(series, desired_value)
print(percentile_rank)





# Handle cases where clicks are 0 to avoid division by zero errors
df['column'] = df['column'].replace([float('inf'), -float('inf')], 0).fillna(0)



df_churned['Age at Churn'] = (df_churned['Churn Date'] - df_churned['Date of Birth']).astype('timedelta64[Y]')

coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)



df['Body'] = df['Body'].str.translate(str.maketrans('', '', string.punctuation))


df_result = df['customerEmailSentFrom'].value_counts().to_frame(name='frequency')



df_agg = df_agg.sort_values(by=['nunique', 'Country'], ascending=[False, True])

df_text['MESSAGE'] = df_text['MESSAGE'].astype(str).str.translate(str.maketrans('', '', string.punctuation))



text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation



# Remove HTML tags
soup = BeautifulSoup(text, 'html.parser')
text = soup.get_text()

# Remove non-alphanumeric  
characters except spaces
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)



>>> df.head()
   YEAR    JAN    FEB    MAR    APR    MAY    JUN    JUL  ...    OCT    NOV    DEC  D-J-F  M-A-M  J-J-A  S-O-N  metANN
0  1910  27.29  26.99  26.49  26.19  27.19  27.49  27.69  ...  28.29  28.29  27.79  27.33  26.62  27.72  28.52   27.55

>>> df_monthly = df.melt(id_vars=['YEAR'], value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], var_name='Month', value_name='Temperature')
>>> df_monthly
      YEAR Month  Temperature
0     1910   JAN        27.29
1     1911   JAN        26.99


>>> df.head()
| Month               |   Actual |   Forecast |
|:--------------------|---------:|-----------:|
| 1996-12-01 00:00:00 |    28.29 |    89.1623 |
| 1997-12-01 00:00:00 |   999.9  |    97.6745 |

>>> df.reset_index().melt('Month', var_name='category', value_name='temperature')
|    | Month               | category   |   temperature |
|---:|:--------------------|:-----------|--------------:|
|  0 | 1996-12-01 00:00:00 | Actual     |         28.29 |
|  1 | 1997-12-01 00:00:00 | Actual     |        999.9  |



df.boxplot(column=[col], by='cluster_group')


df.columns[df.columns.str.contains('hour|day|month|year')]



# Explain /class_weight='balanced'/ in the following expression:
# model = RandomForestClassifier(class_weight='balanced', random_state=42)
# In the expression model = RandomForestClassifier(class_weight='balanced', random_state=42), the parameter class_weight='balanced' is used to address the issue of imbalanced datasets in machine learning classification tasks. class_weight='balanced' tells the Random Forest Classifier to automatically adjust the weights assigned to each class during training. It calculates these weights inversely proportional to the class frequencies. In simpler terms, the minority classes are given higher weights, while the majority classes are given lower weights.



is_all_values_of_X_exist_in_Y = set(X).issubset(Y)




from scipy.stats import boxcox

# Apply Box-Cox transformation to `Price`
price_boxcox, price_lambda = boxcox(df['Price'])

#Convert the predicted and actual prices back to their original scale
y_pred_original = inv_boxcox(y_pred, price_lambda)



# Filter columns starting with digit or '/'
df_dates = df.filter(regex='^\\d|/')


# /X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)/
# By default, train_test_split shuffles the data randomly before splitting. This is important to ensure that the training and testing sets are representative of the overall data distribution and to prevent any biases that might arise from the order of the data. With shuffle=False the data will be split into training and testing sets sequentially, maintaining the original order.



# If you have a DataFrame with 5 columns and you set thresh=3, it translates to:
#     Keep rows that have at least 3 non-NaN values
#     Remove rows that have more than 2 NaNs (5 columns - 3 = 2)
sample_df.dropna(thresh=df.shape[1] - 2, inplace=True)


# Calculate 9:00 AM today
nine_am_today = datetime.datetime.today().replace(hour=9, minute=0, second=0, microsecond=0)



# Use lambda to add a new column using still not defined columns.
df = pd.DataFrame({
    'Unit price': np.random.normal(loc=15.0, scale=5.0, size=100),  # Mean price: 15, Std: 5
    'Quantity': np.random.randint(1, 10, size=100),  # Random quantities from 1 to 9
    'Tax 5%': lambda x: x['Unit price'] * x['Quantity'] * 0.05,  # Tax is 5% of total
})



'Critic_Score': np.clip(np.random.normal(loc=75.8, scale=10.8, size=N), 0, 100),  # Clip to 0-100



humidity = np.random.normal(loc=60, scale=10, size=12).clip(min=0, max=100)  # Humidity in percentage






