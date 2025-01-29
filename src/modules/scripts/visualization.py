import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scripts.utils import get_group_index_by_signal_index
from matplotlib.lines import Line2D
from colorsys import rgb_to_hls, hls_to_rgb
from colorama import Fore, Style
from scripts.config import work_dir

def log(title: str, message: str = '', emphasize: bool = False) -> None:
    if message != '':
        dashes = ""
        for i in range(50 - (18+len(title)) ):
                dashes += "-"
        print("\n------------------", Fore.CYAN + title + Style.RESET_ALL, dashes)
        print(message)
        print()
    elif emphasize:
        print()
        print('-> ', Fore.CYAN + title + Style.RESET_ALL)
        print()
    else:
        print('\n-> ' + title )


def adjust_color(r, g, b, factor):
    h, l, s = rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls_to_rgb(h, l, s)
    
    return '#{:02x}{:02x}{:02x}'.format(*(int(r * 255), int(g * 255), int(b * 255)))
    
def lighten_color(r, g, b, factor=0.1):
    return adjust_color(r, g, b, 1 + factor)
    
def darken_color(r, g, b, factor=0.1):
    return adjust_color(r, g, b, 1 - factor)


def visualize_signals(original, predicted =pd.DataFrame(), detection_values = None, filename="", columns = 5, num_signals = 20, config = None, debug_info=False, save = False):
        
        # ---------------------- Prepare data for visualisation ------------------

        # SIGNAL NAMES ---------
        # Extract signal names from groups for titles of plots, if config provided
        if config is not None:
                # get signal_ids which contains the groups of signals
                signal_ids = config.get('signal_ids')
                flat_signal_ids = []
                for key in signal_ids:
                       flat_signal_ids.append(signal_ids[key])

        # COLUMN SIZE ---------
        # set number of signals in a row (column size)
        if columns == 5: # if default, set to smaller if there are not enough signals to plot
                columns = 5 if num_signals > 5 else 2
        if debug_info: print(columns, "cols")

        # VALUES ---------
        # extract values from the dataframe containing original signal values
        orig_values = original.values
        # if we got prediction as well, extract those values too
        if not predicted.empty:
                pred_values = predicted.values
        # if not, print some debug info
        else:
                if debug_info: print("Only original dataframe was given")

        # ROW SIZE ---------
        # calculate how many rows do we need to plot
        # if there is a remainder, second one is True, which is 1 if added
        rows = int(original.columns.size / columns) + (original.columns.size % columns > 0) 


        # ---------------------- Colors for detection ------------------
        colors_grouped = []
        indexes_grouped = []

        if detection_values is not None:
                
                # detection_values is a dict, storing dicts for every group, keys are group indexes (1,2,3...), then keys are time indexes (1...17500), values are percentages of loss values higher than threshold ... maybe overcomplicated
                max_percentage = max( [ detection_values[4][i] for i in detection_values[4] ] ) # TODO: THIS IS NOT GOOD, SHOULD SEARCH ALL GROUPS, NOW I KNOW THAT THE 4. GROUP SHOULD CONTAIN THE HIGHEST LOSS, BUT OTHERWISE NO
                #max_percentage = max([i for group in detection_values for i in group if i != 0]) 
                min_anomaly = min( [detection_values[group_idx][i] for group_idx in detection_values for i in detection_values[group_idx] ])

                # iterate groups of detection values
                for group_idx in detection_values:
                        
                        # this will contain the detection values in the current group (it is only 1d array, detection is for all the signals, no multiple detection in a group)
                        group = detection_values[group_idx]
                       
                        colors = []
                        x = []

                        # iterate over the detection values in time
                        for index in group:
                                
                                y = group[index]
                                # if we detected something
                                if y != 0:
                                        # store index of detection
                                        x.append(index)
                                        
                                        # TODO: this section should be outside the loop, but i don't want to test this now
                                        move_this_outside_the_loop = True
                                        if move_this_outside_the_loop:
                                                # get color map from yellow to red
                                                cmap = matplotlib.cm.get_cmap('YlOrRd')
                                                #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["springgreen","greenyellow","yellow","darkorange","orangered"])
                                                # normalizer between min and max
                                                norm = plt.Normalize(min_anomaly, max_percentage)
                                                
                                        # normalize current detection value and get a color from colormap
                                        color = cmap(norm(y))

                                        #store color
                                        colors.append(color)
                                else:
                                       raise Warning("Y values should be > 1, because these are percentages of loss values that are higher than threshold")
                        # store colors for group
                        colors_grouped.append(colors)
                        # store indexes for group
                        indexes_grouped.append(x)

        if debug_info:
               print([len(g) for g in colors_grouped])

        # deal with plotting if there is only one signal 
        if num_signals == 1:
                #plot original
                plt.figure(figsize=(18, 6))
                plt.plot( orig_values, color='g', label="original signal")
                #if there is prediction, plot it
                if not predicted.empty:
                        plt.plot(pred_values, color='b', label="reconstruction")
                #NOTE: no detection this case?
        
        # deal with plotting if there are multiple signals, and we need columns and rows
        else: 
                # size figure and make subplots
                fig, ax = plt.subplots(rows, columns, figsize=(30,5*rows))

                # counters for rows and columns
                r = 0
                c = 0
                # iterate over each signal individually
                for i in range(original.shape[1]):

                        # deal with plotting if there are multiple rows
                        if rows != 1:
                                #if last column, c counter to zero (column), increment r counter (row)
                                if c == columns:
                                        c = 0
                                        r += 1

                                # plot original on the current subplot
                                ax[r, c].plot(orig_values[:,i], color='tab:blue', label="original")

                                # if we got predictions, plot this on same subplot
                                if not predicted.empty:
                                        ax[r, c].plot(pred_values[:,i], color='r', label="reconstruction")
                                
                                # set legend for labels
                                ax[r, c].legend(loc='upper left', fontsize="15")
                                # set title with the name of current signal
                                ax[r, c].set_title('_signal_'+str(original.columns[i]))

                                # if we got detections as well, plot them
                                if detection_values is not None:
                                        
                                        # get the index of the group of the current signal (for example this is the 4th signal from the 2nd group)
                                        group_index = get_group_index_by_signal_index(i, config)
                                        # get iterator for the appropriate group-colors we stored earlier
                                        color_iter = iter(colors_grouped[group_index])
                                        
                                        # iterate over the array of detections where we actually detected something (stored earlier)
                                        for index in indexes_grouped[group_index]:
                                                # get the next color #TODO: is this now the second color for the first value? should be after the plotting
                                                current_color = next(color_iter)
                                                # plot vertical line at this index, where the detection happend, with the color of the intensity
                                                ax[r, c].axvspan(index, index+1, ymin=0, ymax=1, alpha=1, color=current_color)
   
                                        # print something to know where we are in the process
                                        log('signal ' + str(i))
                                        
                                # save figure if we got filename
                                #if filename != "": 
                                #        plt.savefig(str(filename)[:-4]+'_signal_'+str(original.columns[i])+'.png')

                                # increment the column counter
                                c += 1
                        
                        #if we only have 1 row, so 2 signals
                        else:

                                if c == columns:
                                        c = 0
                                        r += 1

                                ax[ c].plot( orig_values[:,i], color='b', label="original signal")
                                

                                if not predicted.empty:
                                        ax[ c].plot(pred_values[:,i], color='r', label="reconstruction")
                                
                                
                                ax[ c].set_title('_signal_'+str(original.columns[i]))
                                
                                #if filename != "": 
                                #        plt.savefig(str(filename)[:-4]+'_signal_'+str(original.columns[i])+'.png')

                                c += 1

                # allign plots a bit
                fig.tight_layout(pad=2.0)
        
        # show plots
        plt.show()

        if save:
               filename = "temp" if filename == "" else filename
               plt.savefig(f"{work_dir}/plots/{filename}.png")
        



import numpy as np 
from random import randint
def visualize_signals_grouped(original, correlation_groups, predicted =pd.DataFrame(), savefig = False, filename="", columns = 5, num_signals = 20, debug_info=False):
        
        colors = ['#7b241c', '#884ea0', '#008800', '#ff0027', '#0018a0', '#0cf5c6', '#1d8348', '#f1c40f', '#e95bff', '#34495e']
                        
        colors_dictionary = {}
        for group in correlation_groups:
                if group not in colors_dictionary.keys():
                        colors_dictionary[group] = colors.pop(0)
                        colors.append(colors_dictionary[group])



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

                        color = colors_dictionary[correlation_groups[i]]

                        if c == columns:
                                c = 0
                                r += 1

                        ax[r, c].plot( orig_values[:,i], color=color, label="original signal")
                        ax[r, c].set_facecolor(color+'40')
                        if not predicted.empty:
                                ax[r, c].plot(pred_values[:,i], color='r', label="reconstruction")
                        

                        
                        ax[r, c].set_title('_signal_'+str(original.columns[i]))
                        
                        if savefig: 
                                plt.savefig(str(filename)[:-4]+'_signal_'+str(original.columns[i])+'.png')

                        c += 1
                fig.tight_layout(pad=2.0)
        plt.show()

from colorama import Fore, Back, Style
def visualize_config(config, trained=False, less=False):

        #train_path = config["train_folder"]
        time_steps = config["window_size"]
        multistep = config["multistep"]
        description = config["description"]
        signal_ids_dict = config["signal_groups"]
        stride = config["stride"]
        signal_ids = [signal_ids_dict[key]["signals"] for key in signal_ids_dict]
        units =  [signal_ids_dict[key]["units"] for key in signal_ids_dict]
        num_signals = sum([len(group) for group in signal_ids])

        if trained:
                stopped_at_epoch = config["stopped_at_epoch"]
                type = config["type"]
                print(Fore.CYAN + 'Model type was: ' + Style.RESET_ALL ,type)
                print(Fore.CYAN + 'with filter number ' + Style.RESET_ALL , units)
                print(Fore.CYAN + 'sliding window was ' + Style.RESET_ALL , time_steps)
                print(Fore.CYAN + 'with stride ' + Style.RESET_ALL , stride)
                print(Fore.CYAN + 'training stopped after ' + Style.RESET_ALL , stopped_at_epoch , " epoch")

        if not less:
                log("Config note: ", description)
                #log("Files", f"Train: '{train_path}'")
                log("Data parameters", str(num_signals) + " signals: " + str(signal_ids) + "\nForecasting " + str(multistep) + " target values" + " \nFrom " + str(time_steps) + " past values (sliding window)")


def plot_evaluation( ax_, threshold, true_anomaly_indexes, group_of_detection_indexes, max_value, originals, predictions ):

    # colors for plotting
    original_color = 'teal'
    prediction_color = 'mediumblue'
    true_attack_color = 'lime'
    detection_color = 'sandybrown'
    threshold_color = 'tomato'
    
    ax_.vlines(x=true_anomaly_indexes, ymin=0, ymax=max_value, colors=true_attack_color)
    ax_.vlines(x=group_of_detection_indexes, ymin=0, ymax=max_value, colors=detection_color, alpha=0.5)
    ax_.axhline(y=threshold, alpha=0.8, color=threshold_color, linestyle='--')

    ax_.plot(originals, color=original_color)
    ax_.plot(predictions, color=prediction_color)
    
    # set labels for the plot
    custom_lines = [Line2D([0], [0], color=original_color, label = 'original', lw=2),
                    Line2D([0], [0], color=prediction_color, label = 'prediction', lw=2),
                    Line2D([0], [0], color=threshold_color, label="threshold", linestyle='--', lw=2),
                    Line2D([0], [0], color=true_attack_color, label = 'true attack', lw=2, marker='|', linestyle='None', markersize=10, markeredgewidth=2),
                    Line2D([0], [0], color=detection_color, label = 'detection', lw=2, marker='|', linestyle='None', markersize=10, markeredgewidth=2)]
    ax_.legend(handles=custom_lines, loc='upper left')