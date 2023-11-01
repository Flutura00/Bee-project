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
import seaborn as sns


year = 2019
import os

os.chdir('/sharedcodes/bees/code/')

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



def get_color_spectrum(num_colors,palette):
    listcolors = []
    for i in sns.color_palette(palette,num_colors):
        listcolors.append(i)
    return listcolors
        
        
def function_to_control_for_camera_trajetories(df_cam):
    condition = df_cam['camera'] == 1
    df_cam['x'] = np.where(condition, df_cam['x'] +3000, df_cam['x'])
    return df_cam


def calculate_window_averages(data, window_size,cut_smaller_than,cut_bigger_than):
    windows = []
    num_windows = len(data) - window_size + 1
    for i in range(num_windows):
        window = data[i:i+window_size]
        filtered_values = [val for val in window if val > cut_smaller_than and val <= cut_bigger_than]
        if len(filtered_values) == 0:
            window_avg = 0
        else:
            window_avg = sum(filtered_values) / len(filtered_values)
        windows.append(window_avg)
    windows = windows + ([0] * (window_size - 1))
    return windows

def get_and_inspect_windows(data, window_size,cut_smaller_than,cut_bigger_than):
    window_avg_ls = []
    window_std_ls = []
    window_sem_ls = []
    len_after_0 = []
    num_windows = len(data) - window_size + 1
    windows = []
    min_vals = []
    max_vals = []
    range_vals = []
    how_many_10_range_of_max_ls = []
    which_10_range_of_max_ls = []
    which_window = []
    mean_of_10_max = []
    stdev_of_10_max = []
    window_start_point = []

    for i in range(num_windows): # tqdm
        window = data[i:i+window_size]
        filtered_values = [val for val in window if val > cut_smaller_than and val <= cut_bigger_than]
        if len(filtered_values) == 0:
            window_avg = 0
            window_std = 0
            window_sem = 0
        else:
            window_avg = np.nanmean(filtered_values)
            window_std = np.nanstd(filtered_values)
            window_sem =  window_std / math.sqrt(len(filtered_values))

        #here we inspect the window:
        # range sohuld be 0 to 70 or whatever
        # stdev without 0 values should be extremely love
        window_avg_ls.append(window_avg)
        window_std_ls.append(window_std)
        window_sem_ls.append(window_sem)
        windows.append(window)
        len_after_0.append(len(filtered_values))
        if len(filtered_values)>0:
            min_vals.append(min(filtered_values))
            max_vals.append(max(filtered_values))
            max_val = max(filtered_values)
            how_many_10_range_of_max_vals = [val for val in filtered_values if val > (max_val-10)]
            mean_of_10_max.append(np.mean(how_many_10_range_of_max_vals))
            stdev_of_10_max.append(np.std(how_many_10_range_of_max_vals))

        else:
            min_vals.append(0)
            max_vals.append(0)
            mean_of_10_max.append(0)
            stdev_of_10_max.append(0)
            how_many_10_range_of_max_vals = [0]
        how_many_10_range_of_max_ls.append(len(how_many_10_range_of_max_vals))
        which_10_range_of_max_ls.append(how_many_10_range_of_max_vals)
        which_window.append([i,i+window_size])
        window_start_point.append(i)
        
   # which_10_range_of_max_ls = list(itertools.chain(*which_10_range_of_max_ls)) # first flattenig   
 

    window_avg_ls, window_std_ls, window_sem_ls, len_after_0, windows, min_vals, max_vals, how_many_10_range_of_max_ls, which_10_range_of_max_ls, which_window, mean_of_10_max, stdev_of_10_max,window_start_point =  modify_parameters(window_avg_ls, window_std_ls, window_sem_ls, len_after_0, windows, min_vals, max_vals, how_many_10_range_of_max_ls, which_10_range_of_max_ls, which_window, mean_of_10_max, stdev_of_10_max,window_start_point, window_size = window_size)
    return window_avg_ls, window_std_ls, window_sem_ls, len_after_0, windows, min_vals, max_vals, how_many_10_range_of_max_ls, which_10_range_of_max_ls, which_window, mean_of_10_max, stdev_of_10_max,window_start_point

def modify_parameters(*args,window_size):
    modified_args = []
    for param in args:
        # Modify the value of the parameter without changing its name
        modified_param = param + ([0] * (window_size - 1))
        modified_args.append(modified_param)
    return tuple(modified_args)



def prepare_df_inspect_windows(df,params,param_names):
    for i in range(int(len(params))):
        name_col = param_names[i]
        df[name_col] = params[i]
    return df


def plot_1_bee_given_frames(bee_df,from_time,to_time,color_plot,show_plot):
    while 'level_0' not in bee_df:
        bee_df.reset_index(inplace = True)
    from_time = int(from_time)
    to_time = int(to_time)
    x = bee_df['x'][from_time:to_time].tolist()
    y = bee_df['y'][from_time:to_time].tolist()

    bee_df = bee_df.loc[bee_df['level_0']<=to_time] # majjjsparti e dajna ma pak se, tana ma shume se edhe qe 
    bee_df = bee_df.loc[bee_df['level_0']>=from_time] # majjjsparti e dajna ma pak se, tana ma shume se edhe qe
    plt.plot(x,y, linewidth=0.5, color = color_plot, label = str(from_time)+" - "+str(to_time))
    plt.scatter(x,y,  s=5, color = color_plot)
    plt.gca().set_aspect('equal', adjustable='box')
    if show_plot == True:
        plt.title("time range in frames " +str(from_time) + " - "+ str(to_time))
        plt.show()
    return from_time,to_time

def add_zigzag_timepoints(df, how_many_10_range_of_max_ls, stdev_of_10_max,mean_of_10_max,which_window,how_many_vals_min,stdev_val_max):
    x = 0
    seq = []
    framenums = []
    for i in range(int(len(how_many_10_range_of_max_ls))):
        if ((how_many_10_range_of_max_ls[i]>=how_many_vals_min) and (stdev_of_10_max[i]<=stdev_val_max) and (mean_of_10_max[i]<=120) and (mean_of_10_max[i]>=40)):
            seq.append(1)
        else:
            seq.append(0)
    df['zigzag_value'] = seq
    for i in range(0,len(df)):
        if df['zigzag_value'][i]==1:
            framenums.append(df['framenum'][i])
    frames_hour= [val/10800 for val in framenums ]

    return df, frames_hour


def calculate_clockwise_angles(x_values, y_values):
    num_points = len(x_values)-2
    clockwise_angles = []

    for i in range(num_points):
        x1, y1 = x_values[i], y_values[i]
        x2, y2 = x_values[(i + 1)], y_values[(i + 1)]
        x3, y3 = x_values[(i + 2)], y_values[(i + 2)]

        vector1 = (x1 - x2, y1 - y2)
        vector2 = (x3 - x2, y3 - y2)

        angle_radians = math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0])
        clockwise_angle_degrees = math.degrees(angle_radians)

        if clockwise_angle_degrees < 0:
            clockwise_angle_degrees += 360

        clockwise_angles.append(clockwise_angle_degrees)

    return clockwise_angles

def calculate_segment_length(x1, y1, x2, y2, x3, y3):
    # Calculate the length of each segment
    a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    return a,b,c


#    return distances    

def find_angles_and_consecutive_segments(bee_df):
    
    x1 = np.array(bee_df['x'][0:len(bee_df)-2])
    x2 = np.array(bee_df['x'][1:len(bee_df)-1])
    x3 = np.array(bee_df['x'][2:len(bee_df)])

    y1 = np.array(bee_df['y'][0:len(bee_df)-2])
    y2 = np.array(bee_df['y'][1:len(bee_df)-1])
    y3 = np.array(bee_df['y'][2:len(bee_df)])
    
    a,b,c = calculate_segment_length(x1, y1, x2, y2, x3, y3)

    return a,b,c

def put_values_in_df(bee_df):
    a,b,c = find_angles_and_consecutive_segments(bee_df)
    #bee_df['angle'] = [0] +angles + [0]
    bee_df['distance_x1_x2'] = np.insert(a, [0, len(a)], [0, 0])
    bee_df['segment_x2_x3'] = np.insert(b, [0, len(b)], [0, 0])
    bee_df['segment_x1_x3'] = np.insert(c, [0, len(c)], [0, 0]) 
    return bee_df


def assign_coordinates_to_squares(rect_length, rect_width, total_squares, x_coordinates, y_coordinates):
    # Calculate the size of each square
    square_size = rect_length / np.sqrt(total_squares)

    # Calculate the number of rows and columns in the grid
    num_rows = int(rect_length // square_size)
    num_cols = int(rect_width // square_size)

    # Create a dictionary to store the square assignment for each coordinate
    coordinate_to_square = {}

    # Assign each coordinate to its corresponding square
    for i in range(num_rows):
        for j in range(num_cols):
            square_index = i * num_cols + j
            row_start = i * square_size
            col_start = j * square_size

            # Find the coordinates within the square range
            indices = np.where(
                (x_coordinates >= row_start) & (x_coordinates < row_start + square_size) &
                (y_coordinates >= col_start) & (y_coordinates < col_start + square_size)
            )[0]

            # Assign the coordinates to the square
            for index in indices:
                coordinate_to_square[(x_coordinates[index], y_coordinates[index])] = str(square_index)

    return coordinate_to_square


################################################################################################################## currently not used

def find_continuous_sequences(lst,nr_vals_cut):
    continuous_sequences = []
    start = []
    end = []
    length = []
    start_index = None
    for i, value in enumerate(lst):
        if start_index is None:
            start_index = value
        if i == len(lst) - 1 or value != lst[i + 1] - 1:
            if value > start_index:
                sequence_length = value - start_index + 1
                start.append(start_index)
                end.append(value)
                length.append(sequence_length)
            start_index = None
    merged_sequences = merge_sequences(start, end, length,nr_vals_cut)
    return merged_sequences

def merge_sequences(start, end, length,nr_vals_cut):
    merged_sequences = []
    if len(start) == 0:
        return merged_sequences

    merged_start = start[0]
    merged_end = end[0]
    for i in range(1, len(start)):
        if start[i] - merged_end <= 4:
            merged_end = end[i]
        else:
            merged_sequences.append((merged_start, merged_end, merged_end - merged_start + 1))
            merged_start = start[i]
            merged_end = end[i]
    merged_sequences.append((merged_start, merged_end, merged_end - merged_start + 1))
    return np.array(merged_sequences)

def put_all_together(df,which_bee):
    print("len of the df", len(df))
    bee806= df[df['uid']==which_bee]
    bee806.reset_index(inplace = True, drop = True)
    bee806 = function_to_control_for_camera_trajetories(df_cam = bee806)
    bee806 = put_values_in_df(bee_df = bee806)
    d_v = bee806['distance_x1_x2'].tolist()
    window_avg_ls, window_std_ls, window_sem_ls, len_after_0, windows, min_vals, max_vals, how_many_10_range_of_max_ls, which_10_range_of_max_ls, which_window, mean_of_10_max, stdev_of_10_max,window_start_point  = get_and_inspect_windows(data = d_v, window_size =15,cut_smaller_than = 0,cut_bigger_than = 300)
    params = [window_avg_ls, window_std_ls, window_sem_ls, len_after_0, windows,min_vals, max_vals, how_many_10_range_of_max_ls, which_10_range_of_max_ls, which_window, mean_of_10_max, stdev_of_10_max,window_start_point]
    param_names = ["window_avg_ls", "window_std_ls", "window_sem_ls", "len_after_0", "windows", "min_vals", "max_vals", "how_many_10_range_of_max_ls", "which_10_range_of_max_ls", "which_window", "mean_of_10_max", "stdev_of_10_max","window_start_point"]
    bee_df = prepare_df_inspect_windows(df = bee806,params = params,param_names = param_names)
    bee_df, frames_hour = add_zigzag_timepoints(df= bee_df,how_many_10_range_of_max_ls = how_many_10_range_of_max_ls,stdev_of_10_max =stdev_of_10_max,mean_of_10_max = mean_of_10_max,which_window=which_window , how_many_vals_min = 7,stdev_val_max = 2)
    return bee_df

def hours_zigzag_values(df,day,which_bee):
    hours_of_day = [which_bee,day]
    for i in range(0,24):
        zigzags_this_hr = sum(df.loc[(df['framenum'] > i*10800) & (df['framenum'] <= (i+1)*10800), 'zigzag_value'].values)
        hours_of_day.append(zigzags_this_hr)
    return np.array(hours_of_day)



def prepare_hourly_dataframe_bees(df, nr_bees,day):
    all_bees_all_day = np.tile(np.nan,(nr_bees,26))
    #print(all_bees_all_day)
    df_bees = df['uid'].unique().tolist()  #1710
    for bee in range(0,nr_bees):
        print("bee ", bee)
        which_bee = df_bees[bee]
        df_bee = put_all_together(df = df,which_bee = which_bee)

        hours_of_day =hours_zigzag_values(df = df_bee,day = day,which_bee = which_bee)

        all_bees_all_day[bee] = hours_of_day
      #  plt.rcParams["figure.figsize"] = (10,10)
      #  df_to_plot = df_bee.loc[df_bee['zigzag_value']  == 1]
      #  plt.plot(df_to_plot['x'],df_to_plot['y'])
      #  plt.show()
    cols = ["bee_uid","day",0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    print("cols",len(cols))
    df_day_hour = pd.DataFrame(columns = cols, data = all_bees_all_day)

    return all_bees_all_day


import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(x, y, nr_bins,which_data):
    # Define the fixed coordinate range of 6000 by 6000
    x_min, x_max = 0, 6000
    y_min, y_max = 0, 6000

    # Create a 2D histogram of the coordinates
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=nr_bins, range=[[x_min, x_max], [y_min, y_max]])

    # Plot the heatmap with the specified coordinate range
    plt.imshow(heatmap.T, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='hot')

    # Add the colorbar legend
    plt.colorbar(label='Count')

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'2D Heatmap of X-Y Coordinates {which_data}' )

    # Show the plot
    plt.show()