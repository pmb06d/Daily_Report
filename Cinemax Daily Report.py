# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:22:03 2019

@author: pbonnin
"""

#import pandas as pd
#import time
#import re

# Progress bars!
#from tqdm import tqdm

# To see the current directory
# print(os.getcwd())

# Get all the files in the directory
#from os import listdir
#from os.path import isfile, join

#current_week_number = 38
#current_week = 'Week '+str(current_week_number)
#mypath = '//svrgsursp5/FTP/DOMO/Daily Reports/2019/'+current_week+'/Raw'

# separates the file names and puts them in a list
# 

# Pandas option to display up to 20 columns
#pd.options.display.max_columns = 20
#pd.options.display.max_rows = 100


#%% Function compilation


# gets the previous week number for the autoname functions
def auto_filename():
    import datetime
    today = datetime.date.today()
    lastWeek = today - datetime.timedelta(days=7)
    lastWeek = lastWeek.isocalendar()[1]
    return(lastWeek)
 

# gets an input between 1 and 53 to assign as the week number for the file
def get_filename(): 
    import time
    
    while True:
        week = input('\n'+'Enter the week number in '+str(time.strftime("%Y"))+' for the report you want to run (e.g. 35):')
        try:
            if int(week) <= 53 and int(week) >= 1:
                return(week.title())
                break
            elif week == 'quit':
                break
            else:
                print('\n','Invalid entry',sep='')
            
        except:
            print('\n','Invalid entry',sep='')


# confirms the auto week with the user
def confirm_filename():
    import time
    
    while True:
        input_statement = 'Please confirm Week '+str(auto_filename())+', '+str(time.strftime("%Y"))+' is the correct report (Y/N):'
        get = input(input_statement)
        if get.upper() == 'Y':
            get = 'Week '+str(auto_filename())
            break
        elif get.upper() == 'N':
            while True:
                get = 'Week '+get_filename()
                break
            break
        else:
            print('Please enter only Y or N')
            continue
    return(get.title())


# Gets the path to the folder where the files are hosted
def get_mypath():    
    current_week = confirm_filename()
    mypath = '//svrgsursp5/FTP/DOMO/Daily Reports/2019/'+current_week+'/Raw'
    return current_week, mypath


# Reads in all the text files and compiles them into two dataframes
def data_reader(mypath):
    from os import listdir
    from os.path import isfile, join
    import pandas as pd
    from tqdm import tqdm
    
    # Parse the files to see how to process each one
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    file_list = pd.Series(onlyfiles)
    file_list_clean = file_list.str.replace('.txt','', regex=False).str.replace('DR - ','', regex=False)
    
    region, analysis = [], []
    
    print('\n','Reading in the data:')
    for i in tqdm(list(file_list_clean)):
        temp_list = i.split(' - ')
        region.append(temp_list[0].strip())
        analysis.append(temp_list[1].strip())
        
    files_df = pd.DataFrame(zip(file_list,region,analysis), columns = ['Filename','Country','Analysis'])
        
    channel_rankers = files_df.loc[files_df['Analysis'].str.startswith('Channel'),['Country','Filename']]
    min_by_min = files_df.loc[~(files_df['Analysis'].str.startswith('Channel')),['Country','Filename']]
    
    return channel_rankers, min_by_min


# Skipper function for IBOPE MW output
def skipper(file):
    with open(file) as f:
        lines = f.readlines()
        # get list of all possible lines starting with quotation marks
        num = [i for i, l in enumerate(lines) if l.startswith('"')]
        
        # if not found value return 0 else get first value of list subtracted by 1
        num = 0 if len(num) == 0 else num[0]
        return(num)


# GUI pop-up to select a path     
def get_filepath():
    import tkinter
    from tkinter import filedialog
    
    root = tkinter.Tk()
    root.withdraw()
    return(filedialog.askopenfilename())


# Clean numeric IBOPE variable columns
def clean_features(test_list, cleaned_features=False):
    import re
    
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


# Clean numeric IBOPE variable columns
def to_numeric(feature_name,data):
    import pandas as pd
    data.loc[:,feature_name] = data.loc[:,feature_name].astype(str)
    data.loc[:,feature_name] = data.loc[:,feature_name].str.replace(r'^[Nn]\/*\.*[Aa]\.*[Nn]*$','0', regex=True)
    data.loc[:,feature_name] = data.loc[:,feature_name].str.replace(',','', regex=True)
    return(pd.to_numeric(data.loc[:,feature_name]))


# General pre-processing for channel rankers
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
        

# Target standarizer
def target_normalizer(target_list):
    import re
        
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


# helper function to populate null values on orderd lists
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


# helper function to print the number of unique values per column
def print_unique_count(df):
    for i in list(df):
        print(i+':',len(df[i].unique()))
    

# Compiles all the channel ranker csv's
def compile_rankers(file_path, channel_rankers):
    import pandas as pd
    from tqdm import tqdm
    
    directory = file_path+'/'
    
    df_list = []
    
    print('\n','Compiling channel rankers:')
    for filename, country in tqdm(zip(channel_rankers['Filename'], channel_rankers['Country'])):
        temp_df = pd.read_csv(directory+filename,sep=';',skiprows=skipper(directory+filename),encoding='latin-1')
        temp_df['Region'] = country
        df_list.append(preproc_ranker(temp_df, export_ranking_features=False, convert_date=True))
    
    # stack all the data frames
    channel_rankers_df = pd.concat(df_list, ignore_index=True, sort=False)
    return(channel_rankers_df)


# Pre-process and add a ranking variable
def process_ranker(ranker_df):
    import pandas as pd
    ranking_features, channel_rankers_preproc = preproc_ranker(ranker_df)
    
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
        
    exclude = ['Children','Virtual']
    
    channel_rankers_preproc = channel_rankers_preproc.loc[~(channel_rankers_preproc['Category1'].isin(exclude)),:]
    
    channel_rankers_preproc['Rank_Rat%'] = channel_rankers_preproc.groupby(ranking_features)['Rat%'].rank(ascending=False,method='first')
    
    channel_rankers_preproc.loc[:,'Target'] = channel_rankers_preproc['Target'].apply(target_normalizer)
    
    return(channel_rankers_preproc)


# Uses the zipfile library to extract files from a zip folder
def extract_zip(input_zip):
    import zipfile as zp
    
    input_zip= zp.ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


# Gets the PRG text files from the zip folders
def get_prg_files(current_week, min_by_min):
    import pandas as pd
    import re
    import time
    
    directory = 'R:/Networks Research - PRG Files/'+str(time.strftime("%Y"))
    file = current_week.upper()
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
        
    return list_of_regions, list_of_files


# Parses the PRG files
def parse_prg_txt(file):
    import pandas as pd
    from dateutil import parser
    from datetime import timedelta
    import re
    
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


# Using this dict method speeds up pandas pd.to_datetime significantly
def create_date_dict(date_list):
    import pandas as pd
    from dateutil import parser
    
    # Make sure the passed date list has unique dates
    date_val = [parser.parse(date) for date in date_list]
    date_dict = pd.DataFrame(list(zip(date_list,date_val)))
    date_dict = date_dict.set_index(0).to_dict()[1]
    return(date_dict)


# Applies the PRG parser to all the PRG files
def parse_all_prg_files(regions, files):
    import pandas as pd
    from tqdm import tqdm
    
    list_of_regions = regions
    list_of_files = files
    
    list_of_df = []
    
    print('\n','Parsing PRG files:')
    for region, prg_file in tqdm(zip(list_of_regions,list_of_files)):
        parsed_file = parse_prg_txt(prg_file)
        parsed_file['Region'] = region
        list_of_df.append(parsed_file)
        
    prg_df = pd.concat(list_of_df, sort = False)
        
    prg_df['Insertion'] = prg_df['Description'] +'__'+ prg_df['Date_REF'].map(str) +'__'+ prg_df['Start_time_str'].str.replace('\:00$','',regex=True)
    
    return(prg_df)

# Compiles all the min-by-min text files and returns them in a dataframe
def compile_minute_df(mypath, min_by_min):
    import pandas as pd
    from tqdm import tqdm
    
    temp_list = []
    
    print('\n','Compiling the min-by-min text files:')
    for row in tqdm(range(len(min_by_min))):
        filename = min_by_min.iloc[row,1]
        country = min_by_min.iloc[row,0]
        
        df = pd.read_csv(mypath+'/'+filename, sep = ';',encoding='latin1' )
        df['Region'] = country
        df.columns = clean_features(list(df))
        temp_list.append(df)
    
    min_by_min_df = pd.concat(temp_list, sort = False)

    # clean up the dataset
    
    #the dates
    str_dates = list(min_by_min_df['Date'].unique())
    
    date_dict = create_date_dict(str_dates)
    
    print('\n','Converting dates:')
    tqdm.pandas()
    min_by_min_df['Date_val'] = min_by_min_df['Date'].progress_apply(lambda x: date_dict[x])
    
    # the targets
    target_list = list(min_by_min_df.loc[:,'Target'].unique())
    target_normalizer(target_list)
    
    min_by_min_df.loc[:,'Start Time'] = min_by_min_df['Start Time'].str.replace('\:00$','',regex=True)
    
    print('\n','Normalizing targets:')
    min_by_min_df.loc[:,'Target'] = min_by_min_df['Target'].progress_apply(target_normalizer)
    min_by_min_df.loc[:,'Target'] = min_by_min_df['Target'].str.replace('Personas TV Suscripcion','Pay Universe')
        
    return(min_by_min_df)


# Maps the 5 minute intervals to 1 minute intervals
def get_extrapolator(minute_df):
    import pandas as pd
    
    # read in the single minute reference
    ref = '//svrgsursp5/FTP/DOMO/Daily Reports/one_min_ref.csv'
    minute1 = pd.read_csv(ref)
    minute5 = minute_df['Start Time'].unique()
    minute5.sort()
    minute5 = pd.DataFrame(minute5)
    minute5.columns = ['Start Time']
    minute5['Start Time2'] = minute5['Start Time']
    
    
    extrapolator = minute1.merge(minute5, how='left', left_on='Start Time', right_on='Start Time2')
    extrapolator = extrapolator.loc[:,['TimeBand - 1 minute(s)','Start Time_x','Start Time_y']]
    extrapolator.columns = ['TimeBand - 1 minute(s)','Start Time','Start Time_5mins']
    extrapolator.loc[:,'Start Time_5mins'] = replicate_down(list(extrapolator['Start Time_5mins']))
    return(extrapolator)


# Compiles a dataframe with all the dimensions we need to average rating by PRG
def get_dimensions(min_by_min_df, prg_df):
    import pandas as pd
    import itertools
    
    regional_list = []
    smaller_reg_list = []
    
    for region in list(min_by_min_df['Region'].unique()):
        dimensions = pd.DataFrame(list(itertools.product(list(min_by_min_df.loc[min_by_min_df['Region']==region,'Channel'].unique()),
                                                         list(min_by_min_df.loc[min_by_min_df['Region']==region,'Target'].unique()),
                                                         list(prg_df.loc[prg_df['Region']==region,'Insertion'].unique()))))
    
        loop_dimensions = pd.DataFrame(list(itertools.product(list(min_by_min_df.loc[min_by_min_df['Region']==region,'Channel'].unique()),
                                                         list(min_by_min_df.loc[min_by_min_df['Region']==region,'Target'].unique()),
                                                         list(prg_df['Date'].unique()))))
        dimensions['Region'] = region
        loop_dimensions['Region'] = region
        
        regional_list.append(dimensions)
        smaller_reg_list.append(loop_dimensions)
    
    # compile the detailed dimensions    
    reg_dimensions = pd.concat(regional_list)
    reg_dimensions.columns = ['Channel','Target','Insertion','Region']
    
    dimensions = reg_dimensions.merge(prg_df, how='left', left_on=['Insertion','Region'], right_on=['Insertion','Region'])
    dimensions = dimensions.loc[:,['Region','Target','Channel_x','Description','Date','Start_time_str']]
    dimensions.loc[:,'Start_time_str'] = dimensions['Start_time_str'].str.replace('\:00$','',regex=True)
    
    # compile the looping dimensions
    loop_dims = pd.concat(smaller_reg_list)
    loop_dims.columns = ['Channel','Target','Date','Region']
    
    return dimensions, loop_dims


# Combines the parsed PRG files with the min-by-min dataset through the extrapolator
def prg_rating(region,target,channel,date, data, dimensions, extrapolator):
    
    # This function uses the following datasets:
    ## min_by_min_df <-- holds all the ratings information at a 5 min level
    ## extrapolator <-- maps the 5 mins to 1 min intervals
    ## dimensions <-- holds all the "timebands" (aka programs)
    
    # e.g. This would be a dataframe for Argentina, P18-49, Cinemax, 9/16/2019
    temp = data.loc[(data['Region']==region)&
                    (data['Target']==target)&
                    (data['Channel']==channel)&
                    (data['Date_val']==date),:]
    
    temp = temp.sort_values('Start Time')
    
    # combine with the extrapolator to go from 5 to 1 minute detail
    extra_temp = extrapolator.merge(temp, how='left', left_on='Start Time_5mins', right_on='Start Time')
    extra_temp = extra_temp.loc[:,['Target','Region','Channel','Date_val','Start Time_x','Start Time_5mins','Rat#']]
    
    # 
    dim_temp = extra_temp.merge(dimensions.loc[dimensions['Channel_x']==channel,:], how='left', left_on=['Target','Region','Date_val','Start Time_x',], right_on=['Target','Region','Date','Start_time_str',])
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
    
    return(new_prg_df)


# Loop and get every channel's rating for our prg channel Cinemax using the PRG rating function
def prg_attribution(loop_dims, data, dimensions, extrapolator):
    from tqdm import tqdm
    import pandas as pd
    
    df_stack = []
    
    print('\n','Matching PRG timebands and ratings:')
    for i in tqdm(range(len(loop_dims))):
        r = loop_dims.iloc[i,3]
        t = loop_dims.iloc[i,1]
        c = loop_dims.iloc[i,0]
        d = loop_dims.iloc[i,2]
        df_stack.append(prg_rating(r,t,c,d,data,dimensions, extrapolator))
    
    attributed_prg = pd.concat(df_stack)
    
    return(attributed_prg)


# Cleans the attributed PRG file stack
def CER_prg(attributed_prg):
    import pandas as pd
    
    # CER --> Clean Enhance Rank
    attributed_prg = attributed_prg
    
    # Clean
    attributed_prg.loc[:,'Channel'] = attributed_prg.loc[attributed_prg['Channel'].notna(),]
    attributed_prg.loc[:,'Channel'] = attributed_prg['Channel'].str.replace(' (MF)', '', regex=False)
    attributed_prg.loc[:,'Channel'] = attributed_prg['Channel'].str.replace('_MF', '', regex=False)

    # Enhance
    categories = pd.read_excel('//svrgsursp5/FTP/DOMO/Daily Reports/IBOPE Channel Reference.xlsx')
    attributed_prg = attributed_prg.merge(categories, how='left', left_on='Channel', right_on='MW_Name')

    # check for missing and print out the missing channels
    missing = list(attributed_prg.loc[attributed_prg['Category1'].isna(),'Channel'].unique())
    if len(missing) == 0:
        pass
    else:
        print('There were channels missing in the excel file:')
        [print(i) for i in missing]    


    attributed_prg = attributed_prg.loc[attributed_prg['Category1']!='Children']
    
    prg_ranking_features = ['Target','Region','Description','Date_val','Start_time_str']
    attributed_prg['Rank_Rat#'] = attributed_prg.groupby(prg_ranking_features)['Rat#'].rank(ascending=False,method='first')
    
    keepers = ['Target',
               'Region',
               'Channel',
               'Description',
               'Date_val',
               'Start_time_str',
               'Rat#',
               'Rank_Rat#']
               
    cinemax_attributed_prg = attributed_prg.loc[attributed_prg['Channel']=='Cinemax',keepers]
    return(cinemax_attributed_prg)


def main():
    from os import listdir
    from os.path import isfile, join
    import pandas as pd
    import time
    
# Get the data
    start_time = time.time()
    
    current_week, mypath = get_mypath()
    
    channel_rankers, min_by_min = data_reader(mypath)
    
    
    # Check if the csv exists already in the folder and if there are 7 dates
    # compile the channel rankers and add the ranking variable if not
        
    file_check = [f for f in listdir(mypath.replace('/Raw','')) if isfile(join(mypath.replace('/Raw',''), f))]
    
    if 'Processed Channel Rankers.csv' in file_check:
        avail_dates = pd.read_csv(mypath.replace('Raw','')+'Processed Channel Rankers.csv',usecols = ['Date'])
        avail_dates = list(avail_dates['Date'].unique())
        
        if len(avail_dates) == 7:
            print('\n','Channel rankers already complete for this week')
        else:
            channel_rankers_df = compile_rankers(mypath, channel_rankers)
            channel_rankers = process_ranker(channel_rankers_df)
            cr_output = mypath.replace('/Raw','')+'/Processed Channel Rankers.csv'
            channel_rankers.to_csv(path_or_buf=cr_output, sep=',', index=False)
    else:
        channel_rankers_df = compile_rankers(mypath, channel_rankers)
        channel_rankers = process_ranker(channel_rankers_df)
        cr_output = mypath.replace('/Raw','')+'/Processed Channel Rankers.csv'
        channel_rankers.to_csv(path_or_buf=cr_output, sep=',', index=False)
    
    
    # Check if the attributed prg csv exists already in the folder and if the dates match
        
    if 'Processed PRG files.csv' in file_check:
        avail_dates = pd.read_csv(mypath.replace('Raw','')+'Processed PRG files.csv',usecols = ['Date_val'])
        avail_dates = list(avail_dates['Date_val'].unique())
        
        if len(avail_dates) == 7:
            print('\n','PRG and min-by-min data is already complete for this week')
        else:
            list_of_regions, list_of_files = get_prg_files(current_week, min_by_min)
            
            prg_df = parse_all_prg_files(list_of_regions, list_of_files)
            
            min_by_min_df = compile_minute_df(mypath, min_by_min)
            
            extrapolator = get_extrapolator(min_by_min_df)
            
            dimensions, loop_dims = get_dimensions(min_by_min_df, prg_df)
            
            cinemax_attributed_prg = CER_prg(prg_attribution(loop_dims, min_by_min_df, dimensions, extrapolator))
            
            prg_output = mypath.replace('/Raw','')+'/Processed PRG files.csv'
            
            cinemax_attributed_prg.to_csv(path_or_buf=prg_output, sep=',', index=False)
    
    else:
        list_of_regions, list_of_files = get_prg_files(current_week, min_by_min)
        
        prg_df = parse_all_prg_files(list_of_regions, list_of_files)
        
        min_by_min_df = compile_minute_df(mypath, min_by_min)
        
        extrapolator = get_extrapolator(min_by_min_df)
        
        dimensions, loop_dims = get_dimensions(min_by_min_df, prg_df)
        
        cinemax_attributed_prg = CER_prg(prg_attribution(loop_dims, min_by_min_df, dimensions, extrapolator))
        
        prg_output = mypath.replace('/Raw','')+'/Processed PRG files.csv'
        
        cinemax_attributed_prg.to_csv(path_or_buf=prg_output, sep=',', index=False)
        
    print('\n',"--- %s seconds ---" % (time.time() - start_time))

#%% Test it out (on week 38) 
    
main() 
    
#%% Optional: Create the dynamic ranker

#%% Compile the benchmarks

from datetime import timedelta


date_sets = []
for i in date_val:
    week1 = i - timedelta(days=7)
    week2 = week1 - timedelta(days=7)
    week3 = week2 - timedelta(days=7)
    week4 = week3 - timedelta(days=7)
    week5 = week4 - timedelta(days=7)
    date_sets.append([week1,week2,week3,week4,week5])
    






