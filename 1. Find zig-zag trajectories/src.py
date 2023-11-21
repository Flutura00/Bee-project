
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

def get_xy_day_df(daynum):
    savefile = resultsdir + 'beetrajectories' + '_' + str(daynum).zfill(3) + '.hdf'
    df = pd.read_hdf(savefile)
    day_uids = np.unique(df['uid']).astype(
        int).copy()  # this includes only bees that were alive this day - see '2019 - DBquery'
    day = bd.alldaytimestamps[daynum]
    print('done')
    return df


def frame_to_hour(hour):
    frfr = (int(hour) + ((hour - int(hour)) / 0.6)) * 10800
    tofr = (int(hour + 1) + (((hour + 1) - int(hour + 1)) / 0.6)) * 10800
    return frfr, tofr


def get_xy_hour_and_bees(df, hour, which_bees):
    frfr, tofr = frame_to_hour(hour)
    hour_df = df.loc[(df['framenum'] >= frfr) & (df['framenum'] < tofr)]
    bees_df = hour_df[hour_df['uid'].isin(which_bees)]
    return bees_df


def select_clusters_and_variables_for_zigzags(data_all, df, real_d, i):  # qet zero e ndreq hahahahah
    if real_d == True:
        which_bees = data_all['which_bees_real'][i]
        which_bees = which_bees.strip('][').split(', ')
        which_bees = np.array(which_bees).astype(int)
        mean_sp_90th = data_all['Speed 90th quantile_means'][i]

        hour = data_all['hour'][i]
        day = data_all['day'][i]
        corr = data_all['avg_corr_real'][i]
        len_which_bees = len(which_bees)
    else:
        which_bees = data_all['which_bees_rand'][i]
        which_bees = which_bees.strip('][').split(', ')
        which_bees = np.array(which_bees).astype(int)
        mean_sp_90th = data_all['Speed 90th quantile_means_rand'][i]
        hour = data_all['hour'][i]
        day = data_all['day'][i]
        corr = data_all['avg_corr_rand'][i]
        len_which_bees = len(which_bees)
    df_hour = get_xy_hour_and_bees(df=df, hour=hour, which_bees=which_bees)
    return df_hour, which_bees, [len(which_bees), corr, mean_sp_90th]  #




def save_list_as_csv(data_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_list)


def bin_data(df, bin_size, variable):  # we group by sum! good!
    df = df.set_index(['framenum', 'x', 'y'])
    df = df.groupby(by=[variable]).sum()
    df.reset_index(inplace=True)
    time_index = df[variable]
    df['binned_time'] = time_index - time_index % bin_size + bin_size / 2
    bin_df = df.groupby(
        ['binned_time']).sum()  # for every fish for every stimulus for every second!fatline = sem, and cloud around it
    bin_df.reset_index(inplace=True)
    return bin_df


def plot_heatmap(x, y, nr_bins, which_data):
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
    plt.title(f'2D Heatmap of X-Y Coordinates {which_data}')

    # Show the plot
    plt.show()


def plot_ziggies_day(df, which_bees):
    each_bee_nr_ziggies = []
    print("number of bees ", len(which_bees))
    for bee in range(0, len(which_bees)):  # df_bees):
        df_bee = put_all_together(df=df, which_bee=which_bees[bee])
        nr_ziggies = sum(df_bee["zigzag_value"].tolist())
        each_bee_nr_ziggies.append(nr_ziggies)
    return each_bee_nr_ziggies, df_bee


# i go through 20 clusters and get their correlation and list of zigie values.
# 20 for real and 20 for rand
# now write a function for it here baybe:
def get_some_values(day_df, which_rows, day):
    data_all = pd.read_csv(f'/home/flutura/Files/one_minute_dfs/day_{day}_1_minute.csv')
    nr_vals = 3
    nr_clusters = len(which_rows)
    if len(day_df) == 0:
        df = get_xy_day_df(daynum=day)
    else:
        df = day_df
    data_real_l = np.tile(np.nan, (nr_clusters, nr_vals))  # cols: nr_bees,corr,mean_Val(3), rows = bees_clusters
    data_real_h = np.tile(np.nan, (nr_clusters, nr_vals))

    data_rand_l = np.tile(np.nan, (nr_clusters, nr_vals))  # cols: nr_bees,corr,mean_Val(3), rows = bees_clusters
    data_rand_h = np.tile(np.nan, (nr_clusters, nr_vals))

    ziggies_real_l = []
    ziggies_real_h = []

    ziggies_rand_l = []
    ziggies_rand_h = []
    df_bee_with_zigs_re_h = pd.DataFrame()
    df_bee_with_zigs_re_l = pd.DataFrame()
    df_bee_with_zigs_ra_h = pd.DataFrame()
    df_bee_with_zigs_ra_l = pd.DataFrame()
    for i in tqdm(which_rows):
        temp1, temp2, temp3 = select_clusters_and_variables_for_zigzags(data_all, df, real_d=True, i=i)
        if ((len(temp2) <= 80) and (temp3[1] >= 0.5)):
            df_hour_re, which_bees_re, data_real_h[i] = temp1, temp2, temp3
            each_bee_nr_ziggies_re, df_bee_with_zigs_re_h_temp = plot_ziggies_day(df=df_hour_re,
                                                                                  which_bees=which_bees_re)
            ziggies_real_h.append(each_bee_nr_ziggies_re)
            df_bee_with_zigs_re_h = pd.concat([df_bee_with_zigs_re_h, df_bee_with_zigs_re_h_temp], axis=0)


        elif ((len(temp2) <= 80) and (temp3[1] < 0.5)):
            df_hour_re, which_bees_re, data_real_l[i] = temp1, temp2, temp3
            each_bee_nr_ziggies_re, df_bee_with_zigs_re_l_temp = plot_ziggies_day(df=df_hour_re,
                                                                                  which_bees=which_bees_re)
            ziggies_real_l.append(each_bee_nr_ziggies_re)
            df_bee_with_zigs_re_l = pd.concat([df_bee_with_zigs_re_l, df_bee_with_zigs_re_l_temp], axis=0)

        temp_r1, temp_r2, temp_r3 = select_clusters_and_variables_for_zigzags(data_all, df, real_d=False, i=i)

        if ((len(temp_r2) <= 80) and (temp_r3[1] >= 0.5)):
            df_hour_ra, which_bees_ra, data_rand_h[i] = temp_r1, temp_r2, temp_r3
            each_bee_nr_ziggies_ra, df_bee_with_zigs_ra_h_temp = plot_ziggies_day(df=df_hour_ra,
                                                                                  which_bees=which_bees_ra)
            ziggies_rand_h.append(each_bee_nr_ziggies_ra)
            df_bee_with_zigs_ra_h = pd.concat([df_bee_with_zigs_ra_h, df_bee_with_zigs_ra_h_temp], axis=0)

        elif (len(temp_r2) <= 80 and (temp_r3[1] < 0.5)):
            df_hour_ra, which_bees_ra, data_rand_l[i] = temp_r1, temp_r2, temp_r3
            each_bee_nr_ziggies_ra, df_bee_with_zigs_ra_l_temp = plot_ziggies_day(df=df_hour_ra,
                                                                                  which_bees=which_bees_ra)
            ziggies_rand_l.append(each_bee_nr_ziggies_ra)
            df_bee_with_zigs_ra_l = pd.concat([df_bee_with_zigs_ra_l, df_bee_with_zigs_ra_l_temp], axis=0)

    return ziggies_real_h, ziggies_real_l, ziggies_rand_h, ziggies_rand_l, data_real_h, data_real_l, data_rand_h, data_rand_l, df_bee_with_zigs_re_h, df_bee_with_zigs_re_l, df_bee_with_zigs_ra_h, df_bee_with_zigs_ra_l
