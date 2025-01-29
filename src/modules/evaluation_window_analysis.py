
# This script is for analysing different window sizes for evaluation, it will generate an animation with four plots, showing the effect of choosing different window sizes

# Imports
# For model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tcn import TCN
import tensorflow_addons as tfa
# For data
import pandas as pd
import os
# For visualisation
from scripts.visualization import log, visualize_config
# Config params
from scripts.config import TrainedModels
from scripts.config import work_dir
from scripts.data_preprocess import signal_mask_path
# For evaluation
from scripts.evaluation import multistep_predict
from scripts.data_preprocess import load_data
from scripts.utils import get_name
# For detection
from scripts.evaluation import calculate_group_loss, calculate_areas
from scripts.utils import rolling_median
import numpy as np
import matplotlib.pyplot as plt
# --------------------------- SAVE ANIMATION ----------------------
import matplotlib.animation as animation
from scipy.stats import variation
from scipy import stats
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from functools import partial
import contextlib
# For detection
from scripts.evaluation import calculate_group_loss, calculate_areas, area_detection, read_true_anomalies
from scripts.utils import rolling_median
from matplotlib.lines import Line2D


# basefolder for this script
threshold_window_analysis_dir = f"{work_dir}/evaluation_window_analysis"

# Test GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
log("Num GPUs:" + str(len(physical_devices)))

# ---------------------------  window size range ----------------------------
# set window size
window_size_from = 1
window_size_to = 500
step = 10


# --------------------------- CHOOSE MODEL --------------------------
log("Analysing different window sizes for evaluation", "Read input test files from work_dir/evaluate/input/\nWrites animations to work_dir/evaluate_window_analysis/ model folder/ sub folder")

# ------------  model and model subdir
# read number of model to be used for predictions
model_id = int(input("Enter model id number: "))
model_folder = f"{threshold_window_analysis_dir}/model_{model_id}"
# create model folder if not exists
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# ------------ directory for distributions
losses_dir = f"{model_folder}/losses_with_increasing_window_sizes"
# create model folder if not exists
if not os.path.exists(model_folder):
    os.makedirs(model_folder)


# control debug information
d_info = "no" # input("Evaluate with debug info? (y/n): ")
d_info = True if 'y' in d_info else False

# --------------------------- READ CONFIG AND FILES -----------------
# class to read the model's stored data
tm = TrainedModels()
# read model's config from stored data
model_config = tm.get_data(model_id)
# use new signal mask
signal_mask = signal_mask_path
# visualise config params if needed
if d_info:
    visualize_config(config=model_config)

# --------------------------- LOAD MODEL ----------------------------

# import model h5 file
model_path = work_dir + '/models/' + str(model_id) + '.h5'
model = load_model(model_path, custom_objects={'WeightNormalization':tfa.layers.WeightNormalization, 'TCN': TCN})
# visualise model summary if needed
if d_info:
     model.summary()


# -------------------------------------------------------------------
# Function for anaylsing loss sum distrubution change with different evaluation window sizes
# file_ : trace to predict 
# window_size_from / to : analyse different evaluation window sizes in this range ...
# step: ... with this step between sizes
def evaluate_distribution_file(file_, window_size_from=500, window_size_to=501, step=100):

    # --------------------------- PREDICT ------------------------------

    # get name of potentional prediction
    pred_filename = get_name(type_='predictions', model_id_=model_id, file_=file_, ext_='csv')

    # check if we already have prediction
    if not os.path.isfile(pred_filename):
        log("No saved prediction, prediction now...")
        # Predict with model
        _, df_test,df_pred = multistep_predict(model, model_config, file_, debug_info = d_info, plot = False, signal_mask=signal_mask)
        # store prediction
        df_pred.to_csv(pred_filename, index=False)
    # read prediction if already exists
    else:
        log("Found saved prediction, reading from file.")
        df_test, _ , _ = load_data(file_, model_config, signal_mask=signal_mask)
        df_pred = pd.read_csv(pred_filename)
    # check if we got all signals' prediction
    assert df_pred.shape[1] == 20

    # --------------------------- GROUP LOSSES ------------------------
    log("Calculating group losses...", emphasize = True)

    # Calculate loss of prediction
    group_losses = calculate_group_loss(df_test, df_pred, model_config)

    # Rolling median on losses
    group_losses = rolling_median(group_losses, 20)

    # --------------------------- AREA LOSSES -------------------------

    # array for the different summed losses, with different window sizes
    data = []

    # iterate through each window size
    for window_size in range(window_size_from,window_size_to,step):
        log("Calculating area above threshold with window " + str(window_size))


        # NOTE: remember to change group if needed
        # calculate sum of losses in the window
        areas_grouped = calculate_areas(group_losses[4:5], model_id, window_size=window_size, debug_info = True)

        fixed_values = [] # because somehow these are one element arrays
        
        fixed_values = [loss_value[0] for loss_value in areas_grouped[0] if loss_value != 0] # nullákat lehagyjuk
        #for loss_value in areas_grouped[0]: #because we want the 4th group with the attack
        #    fixed_values.append(loss_value[0] if loss_value != 0 else loss_value)

        data.append(fixed_values) # most ez a negyedik csoport mert csak a 3. és a 4. csoportot kapja meg a calculate_areas

    return data


# -------------------------------------------------------------------
# Function for generate distributions for different evaluation window sizes with evaluate_file() func
# test_files : array of CAN log file paths to generate distribution from
# will save array of distributions as a csv to dist_dir
def generate_distributions():

    # list input files
    test_files = os.listdir(f"{work_dir}/evaluate/input")
    
    # iterate through each test file and analyse different distributions
    for c, file_ in enumerate(test_files):

        # print which test file is being processed
        log(str(c)+". test file", file_)
        
        # get full path of input file
        full_file_ = f"{work_dir}/evaluate/input/{file_}"
        
        # generate different distributions with all window sizes
        # data - will contain all distributions for the given input file
        data = evaluate_distribution_file(full_file_, window_size_from = window_size_from, window_size_to=window_size_to, step=step)
        
        # get name for file
        loss_file_name = get_name(type_="losses_with_increasing_window_sizes", model_id_=model_id, file_=file_, ext_=".", short=True, underscore=False)[:-2]

        loss_file_path = f"{losses_dir}/{loss_file_name}"

        # save array of different distributions
        pd.DataFrame(data).to_csv(loss_file_path, index=False)

# -------------------------------------------------------------------
# Function for evaluating file with current threshold and plot it to the given figure
# same as evaluate.py, but evaluates only attacked group
def plot_evaluation( model_id, ax_, window_size, areas_grouped, threshold, max_value, downsampled_original, downsampled_prediction, anomaly_indexes):

    # colors for plotting
    original_color = 'teal'
    prediction_color = 'mediumblue'
    true_attack_color = 'lime'
    detection_color = 'sandybrown'
    
    ax_.vlines(x=anomaly_indexes, ymin=0, ymax=max_value, colors=true_attack_color)

    detection_indexes_grouped, _ = area_detection(areas_grouped, model_id, entire_array = True, window_size=window_size, threshold = threshold)

    # --------------------------- PLOT ANOMALIES ----------------------
    group = detection_indexes_grouped[0] # now we get back only a 1-element array from area_detection, only the attacked group
    
    # plot detection
    if len(group) > 0:    
        ax_.vlines(x=group, ymin=0, ymax=max_value, colors=detection_color, alpha=0.5)
    
    # original
    ax_.plot(downsampled_original, color=original_color)
    # prediction
    ax_.plot(downsampled_prediction, color=prediction_color)
    
    # set labels for the plot
    custom_lines = [Line2D([0], [0], color=original_color, label = 'original', lw=2),
                    Line2D([0], [0], color=prediction_color, label = 'prediction', lw=2),
                    Line2D([0], [0], color='tomato', label="threshold", linestyle='--', lw=2),
                    Line2D([0], [0], color=true_attack_color, label = 'true attack', lw=2, marker='|', linestyle='None', markersize=10, markeredgewidth=2),
                    Line2D([0], [0], color=detection_color, label = 'detection', lw=2, marker='|', linestyle='None', markersize=10, markeredgewidth=2)]
    ax_.legend(handles=custom_lines, loc='upper left')

def normalize_loss_and_threshold(arr, threshold, t_min, t_max):
    loss_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        loss_arr.append(temp)
    norm_threshold = (((threshold - min(arr))*diff)/diff_arr) + t_min
    return loss_arr, norm_threshold


# -------------------------------------------------------------------
# Function for making animations from saved distributions
#distribution_data_file : generated distribution for a file
# will save animations to output_dir
def make_distribution_animations():

    # ------------ current subdir to use
    # read name of current subdir where animations will be saved
    current_subdir_name = input("Enter name of sub directory, where animations will be stored (under model directory): ")
    current_subdir = f"{model_folder}/{current_subdir_name}"
    # create folder if not exists
    if not os.path.exists(current_subdir):
        os.makedirs(current_subdir)
    
    # ------------ read benign distribution, used for each animation
    S_1_1_benign_loss = "losses_with_increasing_window_sizes_of_model_31_on_S-3-3-benign"
    T_1_1_benign_loss = "losses_with_increasing_window_sizes_of_model_31_on_T-1-1-benign"
    
    benign_loss_path = f"{losses_dir}/{T_1_1_benign_loss}"
    benign_loss = pd.read_csv(benign_loss_path).fillna(0).values # we will get NaN-s at the end ot the arrays, the last values (less than a window size), so we have to fill those with zeros

    # ------------ window range used for distributions
    windows = range(window_size_from, window_size_to, step) # for plot title

    # ------------ functions for animation
    # Initialization function
    def init():

        # clear figure
        ax[0,0].clear()
        ax[0,1].clear()
        ax[1,0].clear()
        ax[1,1].clear()

    # Update function
    def update(frame, ax, ax11_2, max_value, downsampled_original, downsampled_prediction, anomaly_indexes):
        # change NaNs to 0s
        data[frame][np.isnan(data[frame])] = 0

        # set figure title with current window size
        fig.suptitle(f"Evaluation window size is {windows[frame]}", fontsize=12)

        # clear figure
        ax[0,0].clear()
        ax[0,1].clear()
        ax[1,0].clear()
        ax[1,1].clear()
        ax11_2.clear()

        # calculate threholds using benign prediction loss (calculated with the current window size)
        b_loss = benign_loss[frame]
        current_threshold = np.mean(b_loss) + 3 * np.std(b_loss)

        # --------------------- DISTRIBUTION PLOT ------------------------
        bins_count = 100

        # ----- ATTACKED DISTRIBUTION 
        # plot distributions of attacked loss
        ax[0,0].hist(data[frame], bins=bins_count, edgecolor='cornflowerblue',histtype='step',linewidth=1,  alpha=1)
        # get values and bins
        values_a, bins = np.histogram(data[frame], bins=bins_count)
        # get range of bins
        range_from = bins[0]
        range_to = bins[-1]
        # Correct bin edge placement (bins are ranges, for bar plot, we need the center of the bins)
        bins = [(a+bins[i+1])/2.0 for i,a in enumerate(bins[0:-1])]
        # Calculate width of bins
        bin_width = abs(bins[0]-bins[1])

        # ----- BENIGN DISTRIBUTION
        # plot benign distribition in the same range
        ax[0,0].hist(benign_loss[frame], bins=bins_count,  edgecolor='orange',histtype='step',linewidth=1, range=[range_from, range_to],  alpha=1)
        # get values, bins are the same as in attack distribution
        values_b, _ = np.histogram(benign_loss[frame], range=[range_from, range_to], bins=bins_count)

        # ----- DIFFERENCE BETWEEN ATTACKED AND BENIGN DISTRIBUTION
        diff = abs(values_a - values_b)

        # plot bar chart for attacked, benign distribution, and the difference
        ax[0,0].bar(bins, values_a, width=bin_width, color="paleturquoise", alpha=1, label="Attacked loss distribution")
        ax[0,0].bar(bins, values_b, width=bin_width, color="antiquewhite", alpha=1, label="Benign loss distribution")
        ax[0,0].bar(bins, diff, width=bin_width, color="mediumturquoise", alpha=1, edgecolor="lightseagreen", label="Difference")
        #ax[0,0].plot(bins, diff, color="seagreen")

        # set labels
        ax[0,0].set_xlabel("Range of loss values through prediction", fontsize=12)
        ax[0,0].set_ylabel("Count of occurrences", fontsize=12)
        
        # limit axes
        few = 10
        average_of_firts_few_bars = sum(values_a[:few]) / few
        y_lim = average_of_firts_few_bars
        ax[0,0].set_ylim(0, y_lim)

        # plot threshold
        ax[0,0].vlines(x=current_threshold, ymin=0, ymax=y_lim, alpha=0.8, color='tomato', label="threshold",linestyle='--')


        # set title
        name = f"Distribution with window size {windows[frame]}"
        ax[0,0].set_title(name, fontsize=12)
        ax[0,0].legend(loc="upper right")
        

        # ------------------------ DETECTION PLOT -------------------------
        loss_of_group = data[frame]
        loss_of_group = np.append(np.zeros(500), loss_of_group)# pading NOTE: put this into loss calculation!!
        
        # normalise loss and threshold for plotting
        normalized_loss, normalized_threshold = normalize_loss_and_threshold(arr=loss_of_group, threshold = current_threshold, t_min=0, t_max=0.08)
        

        # plot detection
        plot_evaluation(model_id = model_id, ax_=ax[1,0], window_size=windows[frame], areas_grouped=[loss_of_group], threshold=current_threshold, max_value=max_value, downsampled_original = downsampled_original, downsampled_prediction=downsampled_prediction, anomaly_indexes=anomaly_indexes)
            
        # plot loss
        ax[1,0].plot(normalized_loss, color='r', label="loss")
        # plot threshold
        ax[1,0].axhline(y=normalized_threshold, alpha=0.8, color='tomato', linestyle='--')
        ax[1,0].set_title(f"Detection plot", fontsize=12)

        ax[1,0].set_xlabel("Time", fontsize=12)
        ax[1,0].set_ylabel("Value", fontsize=12)
        ax[1,0].set_ylim(bottom=0)

        # --------------------- VARIANCE, KL divergence, J-S distance PLOT ----------------------------
        # colors for plotting
        cv_color = 'deepskyblue'
        kl_color = 'springgreen'
        js_color = 'plum'
        # ------ VARIANCE -------
        # plot variance
        variance_x.append(windows[frame])
        # instead of simple variance, compute the coefficient of variation (defined as the ratio of standard deviation to mean) to be able to compare to other distribution variances
        cv = variation(data[frame])
        variance_y.append(cv)
        cv_description = f"Coefficient of variation is {round(cv,4)}"
        ax[1,1].plot(variance_x, variance_y , color=cv_color)
        ax[1,1].fill_between(variance_x, variance_y, 0, color=cv_color, alpha=.1)

        # ------ KL DIVERGENCE ---
        # Calculating the kl divergence (relative entropy) with scipy
        # define distributions
        p = data[frame]
        q = benign_loss[frame]
        # calculate (P || Q)
        kl_pq = rel_entr(p, q)
        kl_sum = sum(kl_pq)
        kl_y.append(kl_sum)
        kl_description = f"KL(P || Q): {kl_sum:.3f} nats"

        ax11_2.plot(variance_x, kl_y, color=kl_color)
        ax11_2.fill_between(variance_x, kl_y, 0, color=kl_color, alpha=.1)
        ax11_2.set_ylabel("KL divergence")

        # ------ J-S DISTANCE
        # It is more useful as a measure as it provides a smoothed and normalized version of KL divergence, with scores between 0 (identical) and 1 (maximally different), when using the base-2 logarithm.
        # calculate the jensen-shannon distance metric
        # calculate JS(P || Q)
        js_pq = jensenshannon(p, q, base=2)
        # store for later plots as well
        js_y.append(js_pq)
        js_description = f"JS(P || Q) Distance: {js_pq:.3f}"

        ax[1,1].plot(variance_x, js_y , color=js_color)
        ax[1,1].fill_between(variance_x,  js_y, 0, color=js_color, alpha=.1)
       
        ax[1,1].set_ylabel("CV and J-S divergence", fontsize=12)
        ax[1,1].set_xlabel("Increasing window sizes", fontsize=12)
        ax[1,1].set_xlim(0, window_size_to)
        ax[1,1].set_ylim(bottom=0)
        ax[1,1].set_title("CV, KL divergence, and J-S divergence", fontsize=12)

         # set labels for the plot
        custom_lines = [Line2D([0], [0], color=cv_color, label = cv_description, lw=2),
                        Line2D([0], [0], color=kl_color, label = kl_description, lw=2),
                        Line2D([0], [0], color=js_color, label = js_description, lw=2)]
        ax[1,1].legend(handles=custom_lines, loc='upper right')

        
        # --------------------- KOLMOGOROV-SMIRNOV TEST PLOT -------------
        # some part from https://towardsdatascience.com/comparing-sample-distributions-with-the-kolmogorov-smirnov-ks-test-a2292ad6fee5
        # using https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

        loss_a = data[frame]
        loss_b = benign_loss[frame]

        # ----------- KS test ---------------------
        # calculate KS score of two losses (these are two samples, we want to know if they were drawn from the same distribution, if yes, presumably we can't detect, attack looks like noise)
        ks_score = stats.ks_2samp(loss_a, loss_b)

        # add description to the plot, about the KS test
        equal = ks_score[1] > 0.05 # second stored result is the p_value
        description = f"KS statistics of attacked loss vs benign loss\nks = {ks_score[0]:.4f},\np-value = {ks_score[1]:.3e},\nare equal = {equal}"
        color = "antiquewhite" if equal else "mediumturquoise"
        #ax[0,1].set_facecolor(color)
        
        # ----------- Plot test ---------------------
        #sort data
        sorted_a = np.sort(loss_a)
        sorted_b = np.sort(loss_b)

        #calculate CDF values
        cdf_a = 1. * np.arange(len(loss_a)) / (len(loss_a) - 1)
        cdf_b = 1. * np.arange(len(loss_b)) / (len(loss_b) - 1)

        # params for visualising difference
        min_x = max(min(loss_a), min(loss_b))
        max_x = min(max(loss_a), max(loss_b))
        lines_rate = 20

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        # plot difference, lines between the two cdf function
        min_value_in_a = find_nearest(sorted_a, min_x)
        max_value_in_a = find_nearest(sorted_a, max_x)
        from_ = np.where(sorted_a == min_value_in_a)[0][0]
        to_ = np.where(sorted_a == max_value_in_a)[0][0]
        part_ = sorted_a[from_:to_]
        for j in part_[::lines_rate]:

            # current position (on x axes)
            
            # get function value (y axes) of cdf_a in current position
            index = [x for x, val in enumerate(sorted_a) if val >= j][0]
            value_a = cdf_a[index]

            # get function value of cdf_b in current position
            index = [x for x, val in enumerate(sorted_b) if val >= j][0]
            value_b = cdf_b[index]

            # plot line between these points
            ax[0,1].plot([j, j], [value_a, value_b], color = color) #'lightseagreen')

        # plot cdf functions
        ax[0,1].plot(sorted_a, cdf_a, label="CDF of attacked loss")
        ax[0,1].plot(sorted_b, cdf_b, label="CDF of benign loss")

        # set labels for the plot
        custom_lines = [Line2D([0], [0], color="deepskyblue", lw=2),
                        Line2D([0], [0], color="orange", lw=2),
                        Line2D([0], [0], color=color, lw=2)]
        ax[0,1].legend(custom_lines, ['Attacked loss CDF', 'Benign loss CDF', description])

        # x and y labels
        ax[0,1].set_xlabel("Loss values", fontsize=12)
        ax[0,1].set_ylabel("Cumulative probability", fontsize=12)

        # set title of plot
        ax[0,1].set_title("Kolmogorov-Smirnov Test")




        # --------------------- SAVE FIGURE -------------------------------

        # allign plots a bit
        #fig.tight_layout(pad=2.0)
        # save figure
        plt.savefig(f"{frames_dir}/{name}.png") #TODO: make a class, use private frames_dir path

    # ------------ list distribution files
    loss_data_files = [f for f in os.listdir(losses_dir) if 'T-1-1-mal' in f]#os.listdir(losses_dir)

    # iterate through each test file and analyse different distributions
    for c, file_ in enumerate(loss_data_files):

        # print which test file is being processed
        log(str(c)+". losses file", file_)
        # get full path of input file
        full_file_ = f"{losses_dir}/{file_}"
        trace_id = file_[file_.find("on")+3:]
        test_file = f"{trace_id}.log" 

        # read current distribution data
        data = pd.read_csv(full_file_).values

        # for later use, in the update function
        # arrays to store animated variance, kl and js divergence
        variance_x = []
        variance_y = []
        kl_y = []
        js_y = []

        # generate different distributions with all window sizes
        # data - will contain all distributions for the given input file
        #data = evaluate_file(full_file_, window_size_from = window_size_from, window_size_to=window_size_to, step=step)
        
        # get name for file
        loss_name = f"dist_change_{file_[file_.find(str(model_id))+3:]}"
        output_dir = f"{current_subdir}/{loss_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # frames subdir
        frames_dir = f"{output_dir}/frames"
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)


        # Create figure and axis
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        ax11_2=ax[1,1].twinx()
        fig.tight_layout(pad=4.0)


        # --------------------------- plot fix parts --------------------

        test_file_path = f"{work_dir}/evaluate/input/{test_file}"
        pred_filename = get_name(type_='predictions', model_id_=model_id, file_=test_file_path, ext_='csv')
        # Call the function with output suppressed
        with contextlib.redirect_stdout(None):
            df_test, _ , _ = load_data(test_file_path, model_config, signal_mask=signal_mask)
        df_pred = pd.read_csv(pred_filename)
        signal_ids = model_config.get('signal_ids')
        # get max value of data, vertical lines will be this high
        group_index = 5
        max_value = df_test[signal_ids[group_index][0]].max()

        # calculate how much we want to downsample the data
        downsample = len(df_test[signal_ids[group_index]])//50

        downsampled_original = df_test[signal_ids[group_index]][::downsample]
        downsampled_prediction = df_pred[signal_ids[group_index]][::downsample]

        true_anomalies = read_true_anomalies(test_file_path)
        anomaly_indexes = [ind for ind, x in enumerate(true_anomalies) if  x == 1]


        # -------------------------- Create animation
        anim = animation.FuncAnimation(fig, partial(update, ax=ax, ax11_2=ax11_2, max_value=max_value, downsampled_original = downsampled_original, downsampled_prediction = downsampled_prediction, anomaly_indexes = anomaly_indexes), frames=len(data), init_func=init, blit=False)

        # Show the animation
        anim.save(f"{output_dir}/evaluation_windows_animation.mp4")


# -------------------------------------------------------------------

make_distribution_animations()
#generate_distributions()