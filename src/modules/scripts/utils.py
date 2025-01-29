import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scripts.config import work_data

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, 1):
        yield iterable[ndx:min(ndx + n, l)]


def moving_average(grouped_data, average_window_size = 10, plot = False):
    
    ma_grouped = []

    for data in grouped_data:
        moving_averages = []
        for small_window in batch(data, average_window_size):
            # Calculate the average of current window
            window_average = round(sum(small_window) / average_window_size, 8)
                
            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
        
        ma_grouped.append(moving_averages)
        
        if plot:
            plt.figure(figsize=(18, 6))
            plt.plot(data, 'limegreen', label = 'original data', alpha= 1)
            plt.plot(moving_averages, 'mediumvioletred', label = 'moving average')
            plt.legend(loc='upper left', fontsize="10")
            plt.show()
    
    return ma_grouped


def rolling_median(grouped_data, median_window_size = 10, plot = False):
    

    rm_grouped = []

    for data in grouped_data:

        df = pd.DataFrame(data)

        # Calculate the median of current window
        roll_median = df.rolling(window=median_window_size, min_periods=1).median()
        
        rm_grouped.append(roll_median.values)
        
        if plot:
            plt.figure(figsize=(18, 6))
            plt.plot(data, 'limegreen', label = 'original data', alpha= 1)
            plt.plot(roll_median, 'mediumvioletred', label = 'rolling median')
            plt.legend(loc='upper left', fontsize="10")
            plt.show()
    
    return rm_grouped


# calculate cdf function of data and get an x value of an y value (a value of a given percentage)
def cdf_value(data, percentage):

    #sort data
    x = np.sort(data)

    #calculate CDF values
    cdf = 1. * np.arange(len(data)) / (len(data) - 1)

    index = [x for x, val in enumerate(cdf) if val > percentage][0]
    value = float(x[index])

    return value


def flatten(l):
    return [item for sublist in l for item in sublist]

def get_group_index_by_signal_index(signal_index, config):
       signal_ids = config.get('signal_ids')
       count = 0
       for key in signal_ids:
            group_len = len(signal_ids[key])
            count += group_len
            if count > signal_index:
                    return key-1
 
def get_group_index_by_signal_name(signal_name, config):
    signal_ids = config.signal_groups
    count = 0
    for k, group in enumerate(signal_ids):
            group_len = len(group)
            count += group_len
            if signal_name in group:
                    return k

# get a filename to be used for saving
# example: type = 'pred', model_id=1 , file = ...S-1-1-ADD_DEC... -> prediction_of_model_1_on_S_1_1
def get_name(type_, model_id_, file_, ext_, short = False):
    
    trace_name = file_[:-4] # without original extension (.log)

    if "/" in trace_name:
            trace_name = trace_name[trace_name.rfind("/") +1:] # trace name, like T-1-1-malicious... (+1 is for the "/")
    
    name = f"{type_}_of_model_{model_id_}_on_{trace_name}.{ext_}"#.replace('-','_')

    if short:
        return name
    return f"{work_data}/{type_}/{name}"