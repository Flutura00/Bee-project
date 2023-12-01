# Clustering correlation
In this analysis, I cluster the bees into 3 clusters based on the speed 90th quantile, which is the 90th quantile of the speed during 1 minute of acitvity. This analysis reveals that the bees have some degree of synchrony in their intensity of movement withi nthe hive.

This analysis is done in multiple steps, as shown in functions in the functions_for_correlations.py:
1. Extract desired dataframe - We extract fro mthe server, the resolution of data we want to work with. In our case we want to work with 1 minute resolution data.

2. Time subset selection - We subset one day from the database and find out which bees are present during that day. Then obtain a dataframe of those bees in that day.

3. Prepare the clustering and the correlation. We take the parameter we are interested in clustering and run the clustering algorithm fcluster from scipy.cluster.hierarchy,and get the membership of bees in clusters.

4. One day correlation - Then we bring together the correlation matrices with the bee identities, in one array for that day.

5. In parallel, we construct randomized clusters, and calculate their correlation as well.

6. After clustering the bees based on the parameter of interest, in our case the Speed 90th quantile during one minute, we calculate statistics like mean, standard deviation, and correlation of other parameters in the already formed clusters.

7. We bring everything together in a dataframe, in the form of a csv file, to be accessed easliy.


In the end we can simply plot the columns of the new dataframe and find the correlation between different variables in the formed clusters.

Ane example below of the correlation between the speed 90th quantile in a cluster and the number of bees in the corresponding cluster. Speed 90th quantile represents bursts of activity during a 1 minute period.

<img width="620" alt="correlation_bees" src="https://github.com/Flutura00/Bee-project/assets/107845798/55bad3a4-997a-4c8c-aeb5-32667383bb91">
