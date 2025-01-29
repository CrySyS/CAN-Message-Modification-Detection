import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import os
# Clustering
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as spc
import numpy as np
from adjustText import adjust_text 
from matplotlib.colors import ListedColormap
from modules.scripts.config import output_data

from sklearn.manifold import MDS
from modules.scripts.config import Config
from modules.scripts.data_preprocess import load_data
from sklearn.utils import check_random_state

# ------------------------------ Colors and output folder ------------------------------
colors1 = [
       [0.61960784, 0.00392157, 0.25882353, 1.        ],
       [0.84721261, 0.26120723, 0.30519031, 1.        ],
       [0.96378316, 0.47743176, 0.28581315, 1.        ],
       [0.99346405, 0.74771242, 0.43529412, 1.        ],
       [0.74771242, 0.89803922, 0.62745098, 1.        ],
       [0.45305652, 0.78154556, 0.64628989, 1.        ],
       [0.21607074, 0.55563245, 0.73194925, 1.        ],
       [0.36862745, 0.30980392, 0.63529412, 1.        ],
       [0.19607843, 0.05490196, 0.38431373, 1.        ]]

colors2 = [ '#176B87', '#00AD9E', '#55CB92',
          '#A4E57E', '#F9F871', '#F2962F', 
          '#98AFBA', '#CA6F97', 
          '#8671AC', '#9ADFFF', '#308A7C', '#00A4EA']


#cmap = plt.cm.get_cmap('Set1')

# Create the ListedColormap from the colors
cmap = cmap = ListedColormap(colors2)

# output folder where svg figures will be saved
save = False
output_folder = f"{output_data}/please specify a folder name"
output_format = "png"

# ------------------------------ Functions ------------------------------




def analyze_correlation(file, correlation_types):

        
        config = Config("conf1")
        if "SynCAN" in file:
                config = Config("conf3")


        # Read train data from log file, extract signals and scale
        data, _, _ = load_data(file, config)

        if "SynCAN" in file:
                downsampled_data = data.iloc[::200000, :]
                data = downsampled_data


        # Calculate the correlation matrix
        correlation_matrix = np.corrcoef(data.T)

        # Apply MDS to reduce the correlation matrix to 2D with a fixed random seed
        random_state = check_random_state(9)  # Use any integer as the seed
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=random_state)

        signals_2d = mds.fit_transform(1 - correlation_matrix)  # Using 1 - correlation_matrix as a dissimilarity measure

        signal_names = data.columns


        for correlation_type in correlation_types:

                print("############################ Correlation type: ", correlation_type)

                corr_data = data.corr(method=correlation_type)

                # method 1: represent signals with their corelation to other signals, and cluster these "embeddings"
                # method 2: calculate distance from correlation 1-abs(correlation) and cluster  with these distances

                distance_matrix = 1 - abs(corr_data)
                
                
                print("----------------------------- DBSCAN clustering -----------------------------")
                # Perform DBSCAN clustering
                        # method 1
                        # dbscan = DBSCAN(eps=0.5, min_samples=2)  # You can adjust the parameters as needed
                        # correlation_mat_pro = 1 + corr_data
                dbscan = DBSCAN(eps=0.3, min_samples=2, metric="precomputed")  # You can adjust the parameters as needed
                cluster_labels = dbscan.fit_predict(distance_matrix)  # Exclude the first column with names

                print_clusters(data,cluster_labels)

                title  = f"{output_folder}/dbscan_{correlation_type}"
                # plot 2d representation of grouping
                plot_2d_clusters(signals_2d, cluster_labels, signal_names, signals_2d, title = title)
                # plot signals grouped 
                visualize_signals_grouped(original=data, correlation_groups=cluster_labels, title = title)
                



                print("----------------------------- Affinity propagation clustering -----------------------------")
                        # method 1
                        # clustering = AffinityPropagation().fit(corr_data)
                clustering = AffinityPropagation(affinity="precomputed").fit(-distance_matrix)
                # Display the cluster labels
                cluster_labels=clustering.labels_

                
                print_clusters(data,cluster_labels)

                title  = f"{output_folder}/affinity_propagation_{correlation_type}"
                # plot 2d representation of grouping
                plot_2d_clusters(signals_2d, cluster_labels, signal_names, signals_2d, title = title)
                # plot signals grouped 
                visualize_signals_grouped(original=data, correlation_groups=cluster_labels, title = title)
                

                print("----------------------------- Hierarchical clustering with pairwise distances --------------------------")
                        # method 1
                        # pdist = spc.distance.pdist(corr_data.values)
                        # linkage = spc.linkage(pdist, method='complete')
                distance_matrix_1D = squareform(distance_matrix)
                linkage = spc.linkage(distance_matrix_1D, method='complete')
                cluster_labels=spc.fcluster(linkage, 0.2*distance_matrix_1D.max(), 'distance')
                cluster_labels = [x-1 for x in cluster_labels] # To start from 0

                print_clusters(data,cluster_labels)

                title  = f"{output_folder}/hierarchical_{correlation_type}"
                # plot 2d representation of grouping
                plot_2d_clusters(signals_2d, cluster_labels, signal_names, signals_2d, title = title)
                # plot signals grouped 
                visualize_signals_grouped(original=data, correlation_groups=cluster_labels, title = title)
                

                print("----------------------------- MeanShift clustering -----------------------------")
                        # method 1
                        # clustering = MeanShift(bandwidth=0.5).fit(corr_data)
                clustering = MeanShift(bandwidth=0.5).fit(distance_matrix)
                cluster_labels = clustering.labels_

                print_clusters(data,cluster_labels)

                title  = f"{output_folder}/mean_shift_{correlation_type}"
                # plot 2d representation of grouping
                plot_2d_clusters(signals_2d, cluster_labels, signal_names, signals_2d, title = title)
                # plot signals grouped 
                visualize_signals_grouped(original=data, correlation_groups=cluster_labels, title = title)
                






# function to lighten or darken colors
def lighten_color(color, amount=0.5): #https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
        """
        import matplotlib.colors as mc
        import colorsys
        c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        c= colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
        # convert to hsl
        h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
        #s = s * 0.8 
        # set sat to 0.8% or max value
        #s = min(s, 0.4)

        return colorsys.hls_to_rgb(h, l, s)
        
# this function was used to plot heatmaps with different sizes of points, but from the original colab notebook
# from https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
def heatmap(x, y, size):
        # ... used the official colab notebook for heatmaps for each correlation type
        return

# Plot clusters with 2D representation 
def plot_2d_clusters(cluster_data, cluster_labels, signal_names, signals_2d, title):

        # Create a scatter plot of the 2D representation
        plt.figure(figsize=(8, 6))

        # Define a colormap for better visualization of clusters
        colors = cmap(np.linspace(0, 1, len(np.unique(cluster_labels))))

        #print(colors)
        #print(np.unique(cluster_labels))

        for label in np.unique(cluster_labels):
                if label == -1:
                        col = 'k'  # Noise points in black
                else:
                        col = colors[label]

                # Extract the data points for the current cluster label
                cluster_data = signals_2d[cluster_labels == label]

                        # Plot the data points for the current cluster with jitter
                jitter = 0.06  # Adjust the amount of jitter as needed
                x_jittered = cluster_data[:, 0] + np.random.uniform(-jitter, jitter, size=len(cluster_data))
                y_jittered = cluster_data[:, 1] + np.random.uniform(-jitter, jitter, size=len(cluster_data))
                if label == -1:
                        plt.scatter(x_jittered, y_jittered, color=col, label=f'Not clustered', s=100)
                else:
                        plt.scatter(x_jittered, y_jittered, color=col, label=f'Cluster {label}', s=100)


        # Add labels to the points (e.g., signal names) with label collision avoidance
        texts = []
        for i, signal_name in enumerate(signal_names):
                x, y = signals_2d[i, 0], signals_2d[i, 1]
                texts.append(plt.text(x, y, signal_name, alpha=0.7, fontsize=10))

        # Adjust label positions to avoid collisions
        adjust_text(texts, force_points=0.3, force_text=0.3, expand_points=(1, 1))

        plt.title(f"2D Representation of signals {os.path.basename(title)}")
        plt.legend(loc='best')
        # hide ticks
        plt.xticks([])
        plt.yticks([])
        

        if save:
                plt.savefig(f"{title}.{output_format}", format= output_format)
        plt.show()


    


# Plot clusters
def visualize_signals_grouped(original, correlation_groups, title, predicted =pd.DataFrame(), savefig = False, filename="", columns = 5, num_signals = 20, debug_info=False):
        
        colors_list = ['#7b241c', '#884ea0', '#008800', '#ff0027', '#0018a0', '#0cf5c6', '#1d8348', '#f1c40f', '#e95bff', '#34495e']
        # Define a colormap for better visualization of clusters
        colors = cmap(np.linspace(0, 1, len(np.unique(correlation_groups))))
                        
        colors_dictionary = {}
        for group in correlation_groups:
                if group not in colors_dictionary.keys():
                        #colors_dictionary[group] = colors_list.pop(0)
                        #colors_list.append(colors_dictionary[group])

                        c = colors[group]
                        # add only rgb values
                        c = c[:3]
                        colors_dictionary[group] = c



        if columns == 5:
                columns = 5 if num_signals > 5 else 2
        #print(columns, "cols")
        orig_values = original.values
        if not predicted.empty:
                pred_values = predicted.values
        else:
                if debug_info: print("Only original dataframe was given")

        rows = int(original.columns.size / columns) + (original.columns.size % columns > 0) #if there is a remainder, second one is True, which is 1 if added
        
        if num_signals == 1:
                plt.plot( orig_values, color='g', label="original signal")
                if not predicted.empty:
                        plt.plot(pred_values, color='r', label="reconstruction")
        else: 
                fig, ax = plt.subplots(rows, columns, figsize=(30,5*rows))

                r = 0
                c = 0
                for i in range(original.shape[1]):

                        # if correlation group is -1, then it is not in any group
                        if correlation_groups[i] == -1:
                                line_color = '#000000'
                                face_color = '#ffffff'
                        else:
                                color = colors_dictionary[correlation_groups[i]]
                                line_color = '#000000'
                                face_color = lighten_color(color, 0.8)


                        if c == columns:
                                c = 0
                                r += 1


                        ax[r, c].plot( orig_values[:,i], color=line_color, label="original signal", linewidth=2)
                        ax[r, c].set_facecolor(face_color)
                        if not predicted.empty:
                                ax[r, c].plot(pred_values[:,i], color='r', label="reconstruction")
                        

                        
                        ax[r, c].set_title('_signal_'+str(original.columns[i]), fontsize=22)
                        
                        if savefig: 
                                plt.savefig(str(filename)[:-4]+'_signal_'+str(original.columns[i])+'.png')

                        # drop x axis labels
                        ax[r, c].set_xticklabels([])
                        # drop y axis labels
                        ax[r, c].set_yticklabels([])

                        c += 1
                fig.tight_layout(pad=2.0)
        
        if save:
                plt.savefig(f"{title}_signals.{output_format}", format= output_format)
        plt.show()




# from https://medium.datadriveninvestor.com/four-ways-to-cluster-based-on-correlation-a86031adcb4d
# Utility function to print the name of companies with their assigned cluster
def print_clusters(df_combined,cluster_labels):
        cluster_dict = {}
        for i, label in enumerate(cluster_labels):
                if label not in cluster_dict:
                        cluster_dict[label] = []
                cluster_dict[label].append(df_combined.columns[i])

                # Print out the companies in each cluster
        for cluster, companies in cluster_dict.items():
                print(f"Cluster {cluster}: {', '.join(companies)}")



def plot_clusters(data, corr_data):

        print()
        print("Affinity propagation clustering")
        # Perform affinity propagation clustering with default parameters
        clustering = AffinityPropagation().fit(corr_data)
        # Display the cluster labels
        cluster_labels=clustering.labels_
        print_clusters(data,cluster_labels)
        visualize_signals_grouped(original=data, correlation_groups=cluster_labels)

        print()
        print("DBSCAN clustering")
        # Removing negative values in correlation matrix
        correlation_mat_pro = 1 + corr_data
        # Perform DBSCAN clustering with eps=0.5 and min_samples=5, we do NOT have to set the precomputed metric to True, because correlation data is NOT a distance matrix
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(correlation_mat_pro)
        # Print the cluster labels
        cluster_labels=clustering.labels_
        print_clusters(data,cluster_labels)
        visualize_signals_grouped(original=data, correlation_groups=cluster_labels)

        print()
        print("Hierarchical clustering with pairwise distances")
        pdist = spc.distance.pdist(corr_data.values)
        linkage = spc.linkage(pdist, method='complete')
        cluster_labels=spc.fcluster(linkage, 0.2*pdist.max(), 'distance')
        print_clusters(data,cluster_labels)
        visualize_signals_grouped(original=data, correlation_groups=cluster_labels)


