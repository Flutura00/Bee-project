import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import gzip
import os
import gc
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning,message='Mean of empty slice')
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import itertools
import seaborn as sns
import scipy
from tqdm import tqdm
import ast
# x = ast.literal_eval(x)
pd.set_option('display.max_columns', 70)
pd.set_option('display.max_rows', 5)
import statistics as st


year = 2019

%cd '/sharedcodes/bees/'
import os

os.chdir('/sharedcodes/bees/')
# Choose which data I want to work with. There were two experiments, one in 2018 and one in 2019. I work with 2019 data.
if year==2018:
    import definitions_2018 as bd
    resultsdir = '/data/beeresults/'
    comb_contents_dir = '/data/comb-contents-images/'
    zfilln = 2 # for file names
elif year==2019:
    import definitions_2019 as bd
    resultsdir = '/data/beeresults2019/'
    comb_contents_dir = '/data/comb-contents-images2019/'
    zfilln = 3 # for file names    
    
import displayfunctions as bp  # 'bee plots'
import datafunctions as dfunc
dfunc.init(bd) 
bp.init(bd)
bd.year

# Function 1. Call the bee dataset that we are interested in, either in 1 hour resolution, 5 minutes, or 1 minute.
# returns the data in a pandas dataframe
def extract_desired_df(day,unit): # we just take the dataframe we want to work with from that day and resolution
    print(' Real data')
    daystoload = [day]
    numtimedivs = unit
    if numtimedivs==24: # 1 hour resolution data
        prefix = 'dayhour'
    elif numtimedivs==288: # 5 minute resolution data
        prefix = 'day5min'
    elif numtimedivs==1440: # 1 minute resolution data
        prefix = 'day1min'
    else: 
        print('ERROR:  use either numtimesteps == 24 or 288 or 1440')

    for i,daynum in enumerate(daystoload):
        filename = resultsdir+'df_'+prefix+'_'+str(daynum).zfill(zfilln)+'.pklz'
        if i==0:
            [df] = pickle.load(gzip.open(filename,'rb'))
        else:
            df = pd.concat((df,pickle.load(gzip.open(filename,'rb'))[0]))
    return df


# Function 2. Subsets the day we are interested in,and the data of the parameter that we are interested in (ex. speed of the bees) and finds which bees are present during that day. 
# returns the data and the beeID's

def time_subset_selection(df,day,parameter,itera,time_per_plot, shuffle,beelist,unit): # same just for one parameter
    mintimediv = itera*time_per_plot
    maxtimediv = (itera+1)*time_per_plot
    if unit == 24:
        timediv = "Hour"
    else:
        timediv = "timedivision"
        
    interval = int(maxtimediv-mintimediv)
    dfsel =  df.loc[(df['Day number']==day)&(df[timediv]<maxtimediv)&(df[timediv]>=mintimediv)]
    if len(beelist)>0: # allows us, if we want, to select certain bees in our dataframe and leave out the rest?
        dfsel = dfsel.loc[dfsel['Bee unique ID'].isin(beelist)]
                # extra line     dfsel = dfsel[dfsel['Bee unique ID'].isin(cluster_bees)]
    beeids = np.unique(dfsel['Bee unique ID'])
    gb = dfsel.groupby('Bee unique ID')
    data = np.tile(np.nan,(len(beeids),interval)) # I create an array from this data! 
    for i,bee in enumerate(beeids):
        data[i] = gb.get_group(bee)[parameter]
    if shuffle == True: # for the randomized data ,we do the shuffling at this level. 
        for row in data:
            np.random.shuffle(row) 
    return data, beeids


# Function 3. Calculates the correlation and clustering of the bees, clustering is done using fcluster from scipy.cluster.hierarchy 
# Returns the average correlation of each cluster, and the membership, in terms of beeID's in each cluster.
def prepare_correlation(data,n_clusters,nan_to_zero,unit): # nan_to_zero transforms nan values of the MATRIX into 0 if we ask for it.
    normdata = (data-np.nanmean(data))/np.nanstd(data) # we normalize data
    corrmatrix = np.nansum(normdata[:,np.newaxis]*normdata[np.newaxis],axis=-1) # deal with nan values somehow?
#    if ((unit ==24) and (dividents == 24)):
 #       normdata = normdata.flatten()
    corrmatrix = np.corrcoef(normdata) # make correlation matrix. 
#    print("corrmatrix type and value",type(corrmatrix),corrmatrix)
    if nan_to_zero==True:
        if not isinstance(corrmatrix, np.ndarray):
            corr2 = corrmatrix
            corrmatrix = np.array([[0]])
            membership = np.array([[0]])
        else:
            corr2 = np.array([[0]])
            corrmatrix[np.isnan(corrmatrix)] = 0
    
    #data_sorted_by_cluster = np.vstack([data[membership==i] for i in range(n_clusters)]) # WAS IS DAAAS!! this is just data reordered by cluster
    #clusterdivs = np.cumsum([np.sum(membership==i) for i in range(n_clusters)]) #  we are summing up somethingm i think here we do average correlations
    if not isinstance(corr2, np.ndarray):
        avgcorrs_each_cluster = [0]
    else:
        Z = linkage(corrmatrix,method='ward')
        membership = fcluster(Z, n_clusters, criterion='maxclust')-1  # this return cluster memberships labeled as [0,1,... n_clusters-1]? but what is it? alert
        avgcorrs_each_cluster = [np.nanmean(corrmatrix[membership==i][:,membership==i]) for i in range(n_clusters)] # We get the mean of the correlations, list!!!!!!


    return avgcorrs_each_cluster, membership # what comes out of here? Is a list of 3 elements,  each element a list of beeids! so 2 orders. list of lists, where do i get how many in each.### TODO

# Function 3.1
# Function 3 needs and extra funciton in order to return the beeids and the number of bees in each cluster in a cleaner way.
# extra info:
# input : 1. membership is an array with 0,1,2 values for each bee and tells if it belongs to cluster 0,1 or 2 (function nr 3)
#         2. beeids from which you picked to form the clusters in the first place, from the time_subset_selection (function nr 2)
#         3. nr_clusters (function nr 3)

# output  1. which_bees_clusters = a list with as many elements as clusters, and then each element is a list of bees for that cluster!
#         2. a list with as many elements as clusters, with the number of elements for each cluster, which tells us how many bees are in the cluster!

def bee_membership(membership, beeids,n_clusters):       
    membership = np.array(membership)
    result = list(zip(membership, beeids)) 
    result = np.array(result)
    
    which_bees_clusters= [] # which bees belong to each cluster?
    nr_bees = [] # how many bees each cluster
    for char in range(0,n_clusters):
        ex = result[(result[:,0]==char)]
        exls = list(ex[:,1])
        cl = [int(exls) for exls in exls]
        which_bees_clusters.append(cl) # here we extract the ' bees' no need to reorder even ! the clusters, named 0, 1 ,2 . 
        nr_bees.append(len(ex[:,1]))  # order assumet everytime, cl0,cl1,cl2...
    return which_bees_clusters, nr_bees # returns a list of arrays(dimension???) with beeids from each cluster in order 0,1,2..., returns a list of values, how many bees in each cluster, 0,1,2...

#which_bees_clusters, nr_bees = bee_membership(membership, beeids,n_clusters)



# Function nr 4.
# This function brings together the functions defined above, and takes in a dataframe for a day.
# Returns the clusters of that day, for the correlation of the parameter of interest, and beeID's and the number of bees in each cluster.
# Whole day! aLL bees that day! all timechunks that day, in one place! 
# It has the shuffling option that will be used to create a control for the data
def one_day_correlation_real(df,day,parameter,unit,dividents,n_clusters,shuffle,list_of_beelists):
    time_per_plot = unit/dividents # 60 # dividents is 24 
    whole_day_clusters_array = np.tile(np.nan,(dividents,n_clusters))
    whole_day_beeids = np.empty((dividents,n_clusters), dtype=object)

    whole_day_nr_bees = np.tile(np.nan,(dividents,n_clusters))
    for i,itera in enumerate(range(0,dividents)): # iterate through the 24 hours
        # what correlation
        # below, for every iteration, we pick or subset the data we want to work with
        data_this_hour, beeids_this_hour = time_subset_selection(df,day,parameter,itera,time_per_plot,shuffle = shuffle,beelist =([] if len(list_of_beelists) == 0 else list_of_beelists[i]), unit = unit) # data    #   bees_each_hour.append(beeids)'
        # print("shape of data this hour", data_this_hour.shape)
        avgcorrs_all_clusters_this_hour, membership_this_hour = prepare_correlation(data_this_hour,n_clusters,nan_to_zero= True,unit = unit) # cluster divs
        whole_day_clusters_array[i] = avgcorrs_all_clusters_this_hour
        
        # which bees
        which_bees_clusters_this_hour, nr_bees_this_hour =  bee_membership(membership = membership_this_hour, beeids = beeids_this_hour,n_clusters = n_clusters)
        for cl in range(n_clusters):
            whole_day_beeids[i,cl] = which_bees_clusters_this_hour[cl]
        whole_day_nr_bees[i] = nr_bees_this_hour
        
    return whole_day_clusters_array, whole_day_beeids, whole_day_nr_bees

# Function nr 5. This function calls in function 4 in order to create an array of data from the bees and shuffled data for the parameter of interest. The array contians the correlations, the beeids and the number of bees per cluster.
def one_day_reference_parameter_correlations(df, day,parameter,unit,dividents,n_clusters):
    whole_day_clusters_array, whole_day_beeids, whole_day_nr_bees = one_day_correlation_real(df, day, parameter, unit , dividents, n_clusters, shuffle = False,list_of_beelists = [])
    whole_day_clusters_array_rand, whole_day_beeids_rand, whole_day_nr_bees_rand = one_day_correlation_real(df, day, parameter, unit , dividents, n_clusters, shuffle = True,list_of_beelists = [])    
    # this could even come later? the actual names?
    variable_names = ["day","hour","cluster", "nr_bees_real","which_bees_real","avg_corr_real","nr_bees_rand","which_bees_rand","avg_corr_rand"]
    #day_array_all_data = np.tile(np.nan,(dividents*n_clusters,len(variable_names)))
    day_array = [day]*n_clusters*dividents
    values = range(0,dividents)
    hour_array = [val for val in values for _ in range(n_clusters)]
    cluster_array = list(range(1,n_clusters+1))*dividents
    day_array_all_data = np.column_stack((day_array, hour_array, cluster_array, np.ravel(whole_day_nr_bees), np.ravel(whole_day_beeids), np.ravel(whole_day_clusters_array), np.ravel(whole_day_nr_bees_rand), np.ravel(whole_day_beeids_rand), np.ravel(whole_day_clusters_array_rand)))
    return day_array_all_data

#avg

# then corrleation and other values for these clusters are calculated.
# we get an array

# Function nr 6. Now on the clusters that were created based on the reference parameter, we calculate correlation in other parameters.

def one_day_other_parameter_correlation(df,day,parameter,unit,dividents,n_clusters,list_of_beelists): # hour values are day_array_all_data_bla[:.1]
    time_per_plot = unit/dividents
    whole_day_clusters_array = np.tile(np.nan,(dividents,n_clusters))
    means_day  = np.tile(np.nan,(dividents,n_clusters))
    stdevs_day = np.tile(np.nan,(dividents,n_clusters))
    for row in range(0,dividents): # this is how we go through the day, hour by hour! and the function gives us 3 values per hour, three clusters.
        for col in range(0,n_clusters): 
            # what correlation
            i = n_clusters*row+col
            if len(list_of_beelists[i])>0:
                
                data_this_hour, beeids_this_hour = time_subset_selection(df,day,parameter,itera = row,time_per_plot=time_per_plot,shuffle = False,beelist =list_of_beelists[i], unit = unit) # data    #   bees_each_hour.append(beeids)
                # the shape of data_this_hour, is (nr_bees, nr_timepoints_per_plot) # and the nonsense you want to calculate is mean and stdev of the whole box! yes, simple as that :), so i can have for each cluster the mean stdev.
                avgcorrs_all_clusters_this_hour, membership_this_hour = prepare_correlation(data_this_hour,n_clusters = 1,nan_to_zero= True,unit = unit) # cluster divs
                whole_day_clusters_array[row,col] = avgcorrs_all_clusters_this_hour[0]
                # which bees
                which_bees_clusters_this_hour, nr_bees_this_hour =  bee_membership(membership = membership_this_hour, beeids = beeids_this_hour,n_clusters = 1)
                means_day[row,col] =  np.nanmean(data_this_hour)
                stdevs_day[row,col] =  np.nanstd(data_this_hour)

            else:
                whole_day_clusters_array[row,col] = 0
                means_day[row,col] = 0
                stdevs_day[row,col] = 0
    one_day_other_parameters = np.column_stack((np.ravel(whole_day_clusters_array), np.ravel(means_day), np.ravel(stdevs_day)))
    return one_day_other_parameters


# put them together:
# Function nr 7. This function brings together the calling of the dataframe from the server, the clustering based on the reference parameter, and the randomized clusters, and finally loops through the additional parameters to calculate correlation in those parameters for the clusters based on the reference parameter.
def one_day_reference_and_other_params_array(day,parameter_reference,other_params_list, unit,dividents,n_clusters):
    df = extract_desired_df(day,unit)        #                              (df, day, parameter,          unit,dividents,n_clusters)
    day_array_reference_parameter = one_day_reference_parameter_correlations(df, day, parameter_reference,unit,dividents,n_clusters)
    # here 
    for parameter_other in tqdm(other_params_list):
        one_day_other_parameter = one_day_other_parameter_correlation(df, day, parameter_other, unit, dividents, n_clusters, list_of_beelists = day_array_reference_parameter[:,4])
        one_day_other_parameter_rand = one_day_other_parameter_correlation(df, day, parameter_other, unit, dividents, n_clusters, list_of_beelists = day_array_reference_parameter[:,7])
        day_array_reference_parameter = np.column_stack((day_array_reference_parameter, one_day_other_parameter,one_day_other_parameter_rand))
    return day_array_reference_parameter
    
    
    
# Function nr 8. This funciton prepares our variable names for a dataframe that will be created on function 9.
def create_dataframe_columns(other_parameters):
    variable_names = ["day","hour","cluster", "nr_bees_real","which_bees_real","avg_corr_real","nr_bees_rand","which_bees_rand","avg_corr_rand"]
    # we have all the reference variable values + one extra+one extra with rands... 
    values_per_other_param = ["avgcorr","means","stdevs"]
    #other_parameters = ["Speed 90th quantile", "Brood care"]
    additional_variable_names = []
    for parameter in other_parameters:
        additional_variable_names.append([f'{parameter}_{loop}' for loop in values_per_other_param])
        additional_variable_names.append([f'{parameter}_{loop}_rand' for loop in values_per_other_param ])
    additional_variable_names = list(itertools.chain(*additional_variable_names))
    print(additional_variable_names)
    return variable_names + additional_variable_names


# Function nr 9. Brings everything together and returns a dataframe of the clusters average correlation, randomized clusters, and the correlation in other parameters for the data clusters and the randomized clusters.
def create_dataframe_per_day(day,parameter_reference,other_params_list, unit,dividents,n_clusters):
    if unit == 1440:
        minutes = 1
    if unit == 288:
        minutes = 5
    if unit == 24:
        minutes = "24_hrs"
    for i in tqdm(range(0,1)):
        day_array = one_day_reference_and_other_params_array(day,parameter_reference,other_params_list, unit,dividents,n_clusters) # main function
        column_names = create_dataframe_columns(other_parameters = other_params_list)
        dataframe = pd.DataFrame(columns = column_names, data = day_array)
        dataframe.to_csv(f'/home/flutura/Files/day_{day}_{minutes}.csv') # _{n_clusters}_clusters
        print(f"saved df of day {day}")
    #return dataframe
    
    
    




def function_to_control_for_camera_trajetories(df):
    condition = df['camera'] == 1
    df['x'] = np.where(condition, df['x'] +3000, df['x'])
    return df




def plot_1_bee_given_frames(bee_df,from_time,to_time,color_plot,show_plot):
    if 'level_0' not in bee_df:
        bee_df.reset_index(inplace = True)
    from_time = int(from_time)
    to_time = int(to_time)
    x = bee_df['x'][from_time:to_time].tolist()
    y = bee_df['y'][from_time:to_time].tolist()

    bee_df = bee_df.loc[bee_df['level_0']<=to_time] # majjjsparti e dajna ma pak se, tana ma shume se edhe qe 
    bee_df = bee_df.loc[bee_df['level_0']>=from_time] # majjjsparti e dajna ma pak se, tana ma shume se edhe qe 
 #   %matplotlib inline
    
    plt.plot(x,y, linewidth=0.5, color = color_plot, label = str(from_time)+" - "+str(to_time))
    plt.scatter(x,y,  s=5, color = color_plot)
    plt.gca().set_aspect('equal', adjustable='box')
    if show_plot == True:
        plt.title("time range in frames " +str(from_time) + " - "+ str(to_time))
        plt.show()
    return from_time,to_time


def plot_1_bee_given_timepoints(df,beeid,from_time,to_time,plot_color,show_plot):
        
    frfr = (int(from_time) + ((from_time - int(from_time))/0.6))*10800
    tofr = (int(to_time) + ((to_time - int(to_time))/0.6))*10800
    #i want this bee
    name1 = df[df['uid']==beeid].reset_index()
    data_sub2 = name1.loc[name1['framenum']<=tofr] # majjjsparti e dajna ma pak se, tana ma shume se edhe qe 
    #data_sub2 = data_sub2.reset_index()
    data_sub3 = data_sub2.loc[data_sub2['framenum']>=frfr] # majjjsparti e dajna ma pak se, tana ma shume se edhe qe 
    name = data_sub3.reset_index()
    for elem in range(0,len(name)):
        if name['camera'][elem] == 1:
            name.at[elem,'x']= name['x'][elem]+3000 
    
  # %matplotlib inline
    plt.rcParams["figure.figsize"] = (12,12)  
    
    plt.plot(name['x'],name['y'], linewidth=0.5, color = plot_color)
    plt.scatter(name['x'],name['y'],  s=5, color = 'green')
    a = name['x'][0]
    b = name['y'][0]
    plt.scatter(a,b, s=100, color = 'orange')
    plt.gca().set_aspect('equal', adjustable='box')
    if show_plot == True:
        plt.title("time range in hours " +str(from_time) + " - "+ str(to_time))
        plt.show()
    return frfr,tofr



