# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:22:03 2019

@author: pbonnin
"""

import pandas as pd
import time
import tkinter
from tkinter import filedialog
import re

# Progress bars!
from tqdm import tqdm

# To see the current directory
# print(os.getcwd())

# Get all the files in the directory
from os import listdir
from os.path import isfile, join

mypath = '//svrgsursp5/FTP/DOMO/Daily Reports/Week 38'

# separates the file names and puts them in a list
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Pandas option to display up to 20 columns
pd.options.display.max_columns = 20
pd.options.display.max_rows = 100

#%% Function compilation

def skipper(file):
    with open(file) as f:
        lines = f.readlines()
        # get list of all possible lines starting with quotation marks
        num = [i for i, l in enumerate(lines) if l.startswith('"')]
        
        # if not found value return 0 else get first value of list subtracted by 1
        num = 0 if len(num) == 0 else num[0]
        return(num)
        
def get_filepath():
    root = tkinter.Tk()
    root.withdraw()
    return(filedialog.askopenfilename())


# a function to clean the numeric features from the csv
def clean_features(test_list, cleaned_features=False):

    temp = []
    temp2 = []
  
    for i in test_list:
        try:
            temp.append(re.findall('^\w+[\%\#]\w*',i)[0])
        except:
            temp.append(i)
      
    for i in test_list:
        if bool(re.match('^\w+[\%\#]\w*',i)) == True:
            temp2.append(re.findall('^\w+[\%\#]\w*',i)[0])
  
    if cleaned_features == True:
        return(temp2, temp)
    else:
        return(temp)


def to_numeric(feature_name,data):
    import pandas as pd
    data.loc[:,feature_name] = data.loc[:,feature_name].astype(str)
    data.loc[:,feature_name] = data.loc[:,feature_name].str.replace(r'^[Nn]\/*\.*[Aa]\.*[Nn]*$','0', regex=True)
    data.loc[:,feature_name] = data.loc[:,feature_name].str.replace(',','', regex=True)
    return(pd.to_numeric(data.loc[:,feature_name]))


def preproc_ranker(a_dataframe, export_ranking_features=True, convert_date=True):
    import pandas as pd
    
    temp_df = a_dataframe.copy()
    
    try:
        temp_df.drop([' '], axis=1, inplace=True)
    except:
        pass
    
    temp_df.loc[:,'Target'] = temp_df['Target'].str.replace('Live / ', '', regex=True)
    temp_df.loc[:,'Channel'] = temp_df['Channel'].str.replace(' (MF)', '', regex=False)
    temp_df.loc[:,'Channel'] = temp_df['Channel'].str.replace('_MF', '', regex=False)
    facts, all_columns = clean_features(list(temp_df), cleaned_features=True)
    temp_df.columns = clean_features(all_columns)
    temp_df = temp_df.dropna(how='all')
    
    for col in facts:
        try:
            temp_df.loc[:,col] = to_numeric(col,temp_df)
        except:
            temp_df.loc[:,col] = to_numeric(col,temp_df.loc[:,col].str.replace('n.a','0', regex=False).replace(',','', regex=False))
            
    for col in all_columns:
        if col == 'Date' and convert_date == True:
            temp_df.loc[:,col] = pd.to_datetime(temp_df.loc[:,col])
        elif col == 'Year':
            temp_df.loc[:,col] = temp_df.loc[:,col].astype('int')
        else:
            continue
    
    info = pd.DataFrame(temp_df.dtypes).reset_index()
    info = list(info.loc[info[0]!='float64','index'])
    info.remove('Channel')
    
    if export_ranking_features == True:
        return(info, temp_df)
    else:
        return(temp_df)
        

# Standarize the targets
def target_normalizer(target_list):
    
    age_regex = re.compile('[PWM][0-9\-\+]+.+')
    age_regex2 = re.compile('\w+\s*\-*Universe+')
    
    if type(target_list) is str:
        try:
            clean_str = age_regex.findall(target_list)[0].replace('04','')
        except:
            try:
                clean_str = (age_regex2.findall(target_list)[0].replace('-',' '))
            except:
                clean_str = target_list
                
        return(clean_str)
    
    else:        
        clean_values = []
        missing = []
            
        for item in target_list:
            try:
                clean_values.append(age_regex.findall(item)[0])
            except:
                try:
                    clean_values.append(age_regex2.findall(item)[0].replace('-',' '))
                except:
                    clean_values.append(item)
                    missing.append(item)
        
        if len(missing)==0:
            return(clean_values)
        else:
            return(clean_values)
            print('Could not normalize these targets:','\n')
            [print(i) for i in missing]


def replicate_down(raw_list):

  column_names = []

  for i in range(len(raw_list)):
    if str(raw_list[i]) == 'nan':
      if str(raw_list[i-1]) == 'nan' and i != 0:
        column_names.append(column_names[-1])
      else:
        column_names.append(raw_list[i-1])
    else:
      column_names.append(raw_list[i])

  return(column_names)
  
#%%  Parse the files to see how to process each one
        
file_list = pd.Series(onlyfiles)
file_list_clean = file_list.str.replace('.txt','', regex=False).str.replace('DR - ','', regex=False)

region, analysis = [], []

for i in list(file_list_clean):
    temp_list = i.split(' - ')
    region.append(temp_list[0].strip())
    analysis.append(temp_list[1].strip())
    
files_df = pd.DataFrame(zip(file_list,region,analysis), columns = ['Filename','Country','Analysis'])
    
channel_rankers = files_df.loc[files_df['Analysis'].str.startswith('Channel'),['Country','Filename']]
min_by_min = files_df.loc[~(files_df['Analysis'].str.startswith('Channel')),['Country','Filename']]      


#%%  Compile the channel rankers and add the ranking variable
    
start_time = time.time()

directory = mypath+'/'

df_list = []

for filename, country in zip(channel_rankers['Filename'], channel_rankers['Country']):
    temp_df = pd.read_csv(directory+filename,sep=';',skiprows=skipper(directory+filename),encoding='latin-1')
    temp_df['Region'] = country
    df_list.append(preproc_ranker(temp_df, export_ranking_features=False, convert_date=True))

# stack all the data frames
channel_rankers_df = pd.concat(df_list, ignore_index=True, sort=False)
ranking_features, channel_rankers_preproc = preproc_ranker(channel_rankers_df)

# add the channel categories
categories = pd.read_excel('//svrgsursp5/FTP/DOMO/Daily Reports/IBOPE Channel Reference.xlsx')
channel_rankers_preproc = channel_rankers_preproc.merge(categories, how='left', left_on='Channel', right_on='MW_Name')

# check for missing
missing = list(channel_rankers_preproc.loc[channel_rankers_preproc['Category1'].isna(),'Channel'].unique())
if len(missing) == 0:
    pass
else:
    print('There are channels missing in the excel file:')
    [print(i) for i in missing]    

#import clipboard
#clipboard.copy(str(list(channel_rankers_full.loc[channel_rankers_full['Category1'].isna(),'Channel'].unique())))

exclude = ['Children','Virtual']

channel_rankers_preproc = channel_rankers_preproc.loc[~(channel_rankers_preproc['Category1'].isin(exclude)),:]


channel_rankers_preproc['Rank_Rat%'] = channel_rankers_preproc.groupby(ranking_features)['Rat%'].rank(ascending=False,method='first')

channel_rankers_preproc.loc[:,'Target'] = channel_rankers_preproc['Target'].apply(target_normalizer)
        
print('Total time:',str(round((time.time() - start_time),4)),'seconds')


#%% Bring in the PRG files

import zipfile as zp

def extract_zip(input_zip):
    input_zip= zp.ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

directory = 'R:/Networks Research - PRG Files/2019'
file = 'WEEK 38'
filename = directory+'/'+file+'.zip'

prg_dict = extract_zip(filename)
cinemax_prgs = [i for i in list(prg_dict.keys()) if i.startswith('Cinemax')]

prg_regions_full = list(min_by_min['Country'].unique())
prg_regions = [i[:3].lower()+'.txt' for i in prg_regions_full]
cinemax_prgs = [i for i in cinemax_prgs if i[-7:] in prg_regions]

region_lookup = pd.DataFrame(list(zip(prg_regions_full, [i.replace('.txt','') for i in prg_regions]))).set_index(1).to_dict()[0]

list_of_regions = [re.compile('[a-z][a-z][a-z](?=\.txt)').findall(i)[0] for i in cinemax_prgs]
list_of_regions = [region_lookup[i] for i in list_of_regions]

list_of_files = []

# this is a bytes object
for file in cinemax_prgs:
    prg_txt = prg_dict[file].decode('latin1').split('\r\n')
    list_of_files.append(prg_txt)

def parse_prg_txt(file):    
    from dateutil import parser
    from datetime import timedelta
    
    numbers = re.compile('(?=\w+\s)[0-9]+')
    title = re.compile('(^\w+\s[0-9\s]+)([\w\,\sÀ-ÿ:\-\#\/]+)(?<!\=\=)')
    # old regex (does not include the "=="
    #extras = re.compile('(?<=\=\=)\w+[\sÀ-ÿ\w]*')
    
    extras = re.compile('\=\=\s*\w*[\sÀ-ÿ\w]*')
    
    channel = []
    date_original = []
    date_value = []
    time_stamp = []
    start_time_int = []
    start_time_str = []
    end_time_int = []
    end_time_str = []
    desc = []
    desc2 = []
    desc3 = []
    desc4 = []
    desc5 = []
    
    for line in file:
        if len(line) != 0:
            # get the channel
            channel.append(re.compile('^\w+').findall(line)[0])
            
            # get the date and time
            date = numbers.findall(line)[0]
            date_original.append(date)
            
            date_val = parser.parse(date)
            date_value.append(date_val)
            
            start = numbers.findall(line)[1]
            
            end = numbers.findall(line)[2]
            
            if int(start[:2]) < 6:
                timestamp = date_val + timedelta(days = 1)
                timestamp = timestamp.replace(hour=int(start[:2]), minute=int(start[2:4]))
                start_time_str.append(str(int(start[:2])+24)+':'+start[2:4]+':'+start[4:6])
                start_time_int.append(int(start)+240000)
                end_time_str.append(str(int(end[:2])+24)+':'+end[2:4]+':'+end[4:6])
                end_time_int.append(int(end)+240000)
                
            else:
                timestamp = date_val.replace(hour=int(start[:2]), minute=int(start[2:4]))  
                start_time_str.append(start[:2]+':'+start[2:4]+':'+start[4:6])
                start_time_int.append(int(start))
                end_time_str.append(end[:2]+':'+end[2:4]+':'+end[4:6])
                end_time_int.append(int(end))
            
            time_stamp.append(timestamp)
            
            # get the title of the program
            desc.append(title.findall(line)[0][1].strip())
    
            # get the extra descriptions
            extra_count = len(extras.findall(line))
            values = [extras.findall(line)[i].strip() for i in range(extra_count)]
            desc2.append(values[0].replace('==',''))
            desc3.append(values[1].replace('==',''))
            desc4.append(values[2].replace('==',''))
            desc5.append(values[3].replace('==',''))
        else:
            continue
    
    column_names = ['Channel', #1
                    'Date_REF', #2
                    'Date', #3
                    'TimeStamp', #4,
                    'Start_time_int', #5
                    'Start_time_str', #6
                    'End_time_int', #7
                    'End_time_str', #8
                    'Description', #9
                    'Desc2', #10
                    'Desc3', #11
                    'Desc4', #12
                    'Desc5'] #13
    
    return(pd.DataFrame(list(zip(channel, #1
                            date_original, #2
                            date_value, #3
                            time_stamp, #4
                            start_time_int, #5
                            start_time_str, #6
                            end_time_int, #7
                            end_time_str, #8
                            desc, #9
                            desc2, #10
                            desc3, #11
                            desc4, #12
                            desc5)), columns = column_names)) #13
    
# test it
#parse_prg_txt(list_of_files[-1]).head()

list_of_df = []

for region, prg_file in zip(list_of_regions,list_of_files):
    parsed_file = parse_prg_txt(prg_file)
    parsed_file['Region'] = region
    list_of_df.append(parsed_file)
    
prg_df = pd.concat(list_of_df, sort = False)
    
prg_df['Insertion'] = prg_df['Description'] +'__'+ prg_df['Date_REF'].map(str) +'__'+ prg_df['Start_time_str'].str.replace('\:00$','',regex=True)

#%% Extrapolate the 5 min file to 1 minute

from dateutil import parser
start_time = time.time()

# compile all the files
temp_list = []

for row in tqdm(range(len(min_by_min))):
    filename = min_by_min.iloc[row,1]
    country = min_by_min.iloc[row,0]
    
    df = pd.read_csv(mypath+'/'+filename, sep = ';',encoding='latin1' )
    df['Region'] = country
    df.columns = clean_features(list(df))
    temp_list.append(df)

min_by_min_df = pd.concat(temp_list, sort = False)
    
print('Total time:',str(round((time.time() - start_time),4)),'seconds')

# clean up the dataset

#the dates
str_dates = list(min_by_min_df['Date'].unique())
date_val = [parser.parse(date) for date in str_dates]
date_dict = pd.DataFrame(list(zip(str_dates,date_val)))
date_dict = date_dict.set_index(0).to_dict()[1]

def clean_with_dict(date,date_dict=date_dict):
    return(date_dict[date])

tqdm.pandas()
min_by_min_df['Date_val'] = min_by_min_df['Date'].progress_apply(clean_with_dict)

# the targets
target_list = list(min_by_min_df.loc[:,'Target'].unique())
target_normalizer(target_list)

min_by_min_df.loc[:,'Start Time'] = min_by_min_df['Start Time'].str.replace('\:00$','',regex=True)
min_by_min_df.loc[:,'Target'] = min_by_min_df['Target'].progress_apply(target_normalizer)
min_by_min_df.loc[:,'Target'] = min_by_min_df['Target'].str.replace('Personas TV Suscripcion','Pay Universe')

# read in the single minute reference
ref = '//svrgsursp5/FTP/DOMO/Daily Reports/one_min_ref.csv'
minute1 = pd.read_csv(ref)
minute5 = min_by_min_df['Start Time'].unique()
minute5.sort()
minute5 = pd.DataFrame(minute5)
minute5.columns = ['Start Time']
minute5['Start Time2'] = minute5['Start Time']


extrapolator = minute1.merge(minute5, how='left', left_on='Start Time', right_on='Start Time2')
extrapolator = extrapolator.loc[:,['TimeBand - 1 minute(s)','Start Time_x','Start Time_y']]
extrapolator.columns = ['TimeBand - 1 minute(s)','Start Time','Start Time_5mins']
extrapolator.loc[:,'Start Time_5mins'] = replicate_down(list(extrapolator['Start Time_5mins']))


min_by_min_df.loc[(min_by_min_df['Target']=='P18-49')&
                  (min_by_min_df['Date']=='Mon Sep 16, 2019')&
                  (min_by_min_df['Channel']=='Cinemax')&
                  (min_by_min_df['Region']=='Argentina'),['Target','Channel','Date_val','Start Time','Rat#']].to_clipboard()

#%% Compile a dataframe with all the dimensions we need to average rating by

import itertools


regional_list = []
for region in list(min_by_min_df['Region'].unique()):
    dimensions = pd.DataFrame(list(itertools.product(list(min_by_min_df.loc[min_by_min_df['Region']==region,'Channel'].unique()),
                                                     list(min_by_min_df.loc[min_by_min_df['Region']==region,'Target'].unique()),
                                                     list(prg_df.loc[prg_df['Region']==region,'Insertion'].unique()))))
    dimensions['Region'] = region
    regional_list.append(dimensions)

reg_dimensions = pd.concat(regional_list)
reg_dimensions.columns = ['Channel','Target','Insertion','Region']

dimensions = reg_dimensions.merge(prg_df, how='left', left_on=['Insertion','Region'], right_on=['Insertion','Region'])
dimensions = dimensions.loc[:,['Region','Target','Channel_x','Description','Date','Start_time_str']]
dimensions.loc[:,'Start_time_str'] = dimensions['Start_time_str'].str.replace('\:00$','',regex=True)


def print_unique_count(df):
    for i in list(df):
        print(i+':',len(df[i].unique()))

# Inisital try with a for loop


#for i in range(len(test)):


start_time = time.time()

i = 0
# e.g. This would be a dataframe for Argentina, P18-49, Cinemax, 9/16/2019
temp = min_by_min_df.loc[(min_by_min_df['Region']==dimensions.iloc[i,0])&
                         (min_by_min_df['Target']==dimensions.iloc[i,1])&
                         (min_by_min_df['Channel']==dimensions.iloc[i,2])&
                         (min_by_min_df['Date_val']==dimensions.iloc[i,4]),:]
temp = temp.sort_values('Start Time')

# combine with the extrapolator to go from 5 to 1 minute detail
extra_temp = extrapolator.merge(temp, how='left', left_on='Start Time_5mins', right_on='Start Time')
extra_temp = extra_temp.loc[:,['Target','Region','Channel','Date_val','Start Time_x','Start Time_5mins','Rat#']]

# 
dim_temp = extra_temp.merge(dimensions.loc[dimensions['Channel_x']==dimensions.iloc[i,2],:], how='left', left_on=['Target','Region','Date_val','Start Time_x',], right_on=['Target','Region','Date','Start_time_str',])
dim_temp = dim_temp.loc[:,['Target',
                           'Region',
                           'Channel',
                           'Description',
                           'Date_val',
                           'Start_time_str',
                           'Start Time_x',
                           'Rat#']]
                           
                           
dim_temp.loc[:,'Description'] = replicate_down(list(dim_temp['Description']))
dim_temp.loc[:,'Start_time_str'] = replicate_down(list(dim_temp['Start_time_str']))


new_prg_df = dim_temp.groupby(['Target', 'Region', 'Channel', 'Description','Date_val','Start_time_str'])['Rat#'].mean()
new_prg_df = new_prg_df.reset_index().sort_values('Start_time_str')
new_prg_df = new_prg_df.reset_index(drop=True)

print('Total time:',str(round((time.time() - start_time),4)),'seconds')