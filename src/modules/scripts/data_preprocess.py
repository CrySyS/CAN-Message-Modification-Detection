import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from bitstring import BitArray
from scripts.config import Config, signal_mask_path, work_dir, signal_mask_path_old_traces
from scripts.config import interesting_columns as interesting_columns_
from scripts.visualization import log
from scripts.visualization import visualize_signals
from scripts.utils import flatten
from pandas.core.frame import DataFrame
from typing import List, Optional, Tuple, Dict

dump_debug_log_to_file = False

zero_positions = {} # dict ( mesage id = vector), vector shows bit positions where value is always 0
one_positions = {} # positions where value is always 1

# Load data from log file
# interpret signals, scale, return in dataframe
def df_from_log(log_file: str, path_signal_mask: Optional[str]=None, minmaxScale: bool=False, raw_data=False, time_as_index = True) -> DataFrame:

        # ------------------------ Setting up signal mask -------------

        if path_signal_mask is None:
                if 'trace' in log_file:
                        path_signal_mask = signal_mask_path_old_traces
                        log("Using modified new signal mask for old traces: " + signal_mask_path_old_traces)
                else:
                        path_signal_mask = signal_mask_path
                        

        elif 'old' in path_signal_mask:
                log("Changing interesting column names back to old ones, due to old signal mask")
                from scripts.config import old_columns
                interesting_columns = old_columns
        else:
                interesting_columns = interesting_columns_

        # --------------------------------------------------------------- #
        # ------------------------ Load dataframe from file ------------- #
        # --------------------------------------------------------------- #
        
        # ------------------------ SynCAN dataset ----------------------

        if "SynCAN" in log_file: 
                # Load the data
                df_temp = pd.read_csv(log_file)
    
                

                # interpolate the data
                df_test = pd.DataFrame()
                
                t = df_temp.index 
                for id in df_temp["ID"].unique():

                        # Filter on message ID
                        filtered = df_temp.query(f'ID=="{id}"')

                        # If there are less than 3 messages, we can't interpolate
                        if len(filtered) < 3:
                                raise Exception("Not enough message to interpolate (probably)")
            
                        # iterate over the 4 signals, and interpolate them, then add them to the final df
                        for i in range(1,5):
                                # get all not nan rows
                                not_nan = filtered[filtered[f"Signal{i}_of_ID"].notna()]
                
                                # if no not nan rows, break
                                if len(not_nan) == 0:
                                        break

                                df_test[f"{id}_{i}"] = np.interp(t, not_nan.index, not_nan[f"Signal{i}_of_ID"])
                
                # rearrange columns will be done on load_data function after returning

                return df_test


        # ------------------------ Normal log file ----------------------
        # By default we want to use log files, but for faster calculations, we may use .csv data already with some preprocess
        elif '.log' in log_file:
                #log("Got log file, ...")
                df_test = pd.read_csv(log_file, header=None, engine='python', dtype=str)
                

                # ------------------------ New log format ----------------------
                # new log format
                if 'can' in str(df_test.iloc[0].values):
                        log("Probably new log files which only have 3 columns")

                        
                        df_test = pd.read_csv(log_file, sep="\s", header=None, engine='python', dtype=str)
                        print(df_test.head())

                        
                        columns = ['TimeStamp', 'can', 'message_and_data']
                        # msg modificiation logs contain an additional two boolean columns
                        if len(df_test.columns) > 3:
                                columns = ['TimeStamp', 'can', 'message_and_data', 'attacked_flag', 'boolean2']
                        # Add column names
                        df_test.columns = columns

                        print(df_test.head())

                        # drop () from timestamp
                        df_test['TimeStamp'] = df_test['TimeStamp'].str[1:-1]

                        # Convert timestamps to float start from 0 and set as index
                        df_test['TimeStamp'] = df_test['TimeStamp'].astype(float)
                        df_test['TimeStamp'] -= df_test['TimeStamp'][0]
                        if time_as_index:
                                df_test = df_test.set_index('TimeStamp')

                        print(df_test.head())

                        # split message id and data into separate columns, ID0 for message ids and DLC_bytes for data
                        message_ids = []
                        data = []
                        for row in df_test['message_and_data']:
                                message_ids.append(row.split('#')[0])
                                data.append(row.split('#')[1])

                        # add new columns
                        df_test['ID0'] = message_ids
                        df_test['DLC_bytes'] = data

                        # ID0 in new format is like 110 and not 0110 which all other code expects, so we add a 0 to the beginning
                        df_test['ID0'] = df_test['ID0'].map(lambda s: '0'+s)

                        print(df_test.head())

                # ------------------------ Old log format ----------------------
                # old log format
                else:
                        # read again with the appropriate separator
                        df_test = pd.read_csv(log_file, sep=' \s+', header=None, engine='python', dtype=str)

                        # Add column names
                        df_test.columns = ['TimeStamp', 'ID0', 'ID1', 'DLC_length', 'DLC_bytes']

                        # Convert timestamps to float start from 0 and set as index
                        df_test['TimeStamp'] = df_test['TimeStamp'].astype(float)
                        df_test['TimeStamp'] -= df_test['TimeStamp'][0]
                        if time_as_index:
                                df_test = df_test.set_index('TimeStamp')
                        
                        
            
                
        # ------------------------ CSV files, not SynCAN ----------------------
        elif '.csv' in log_file:
                log("Got csv file, preprocessing...")
                # Add column names -- DLC = data length code, utolso oszlop pedig a data (utana ket tab elvalaszto karakter majd egy flag, hogy tamadott-e az uzenet)
                df_test = pd.read_csv(log_file, sep=',', header=None, engine='python', dtype=str).dropna(axis=1,how='all')[:-1]
                df_test.columns = ['TimeStamp', 'ID0', 'ID1', 'DLC_length', 'DLC_bytes']

                # Convert timestamps to float start from 0 and set as index
                df_test['TimeStamp'] = df_test['TimeStamp'].astype(float)
                df_test['TimeStamp'] -= df_test['TimeStamp'][0]
                df_test = df_test.set_index('TimeStamp')

        # ------------------------ Everything else, not supported ------------
        else:
                raise ValueError("Bad input file extension")
        

        # -------------------------------------------------------------------- #
        # ------------------------ Preprocessing ----------------------------- #
        # -------------------------------------------------------------------- #

        # Convert DLC_bytes to BitArray
        df_test['DLC_bytes'] = df_test['DLC_bytes'].map(lambda s: BitArray(hex=s))

        if raw_data:
                return df_test


        # ------------------------ Extract signals from log data ------------- 
        # read saved signal mask from file
        store = pd.HDFStore(path_signal_mask)
        signal_mask = store['signal_mask']
        store.close()
        # We will use the signal_mask, so make sure it is set properly and exists
        if signal_mask is None:
                raise ValueError("No signal mask found, because either path_signal_mask is not specified in data_preprocess.py, or no signal mask exists. Use calculate_signals/src/signal_extractor.py and then calculate_signals/signal_mask.py to get a signal mask") 

        # extract signals from can logs with signal mask
        t = df_test.index 
        df = pd.DataFrame() # this df will contain all signals from the logs in separate columns
        for id in signal_mask["id"].unique():
                # Filter on message ID
                filtered: DataFrame = df_test.query(f'ID0=="{id}"')
                if len(filtered) < 3:

                        log(f"Not enough message to interpolate (probably), length of filtered dataframe: {len(filtered)}")
                        log(f"Filtered dataframe: {filtered.to_string()}")
                        log(f"ID: {id}")
                        log(f"File: {log_file}")
                        log(f"Dataframe head: {df_test.head()}")
                        raise Exception("Not enough message to interpolate (probably)")

                # Extract time series using slices
                # iterate over each message id, they will contain multiple signals
                for _, row in signal_mask.query(f'id=="{id}"').iterrows():
                        index = int(row['index'])
                        if index < 0:
                                raise ValueError('Index is ' + str(index) + ", shouldn't be below 0")
                        start = row['start'] # start position of current signal
                        stop = row['stop'] # end position of it
                        part = slice(int(start), int(stop)) # NOTE: is it ok to cast to int? why did signal mask contain 1.0 instead of 1 in the first place? maybe fix it there
                        series: DataFrame = filtered['DLC_bytes'].map(lambda x: x[part].uint) # get this part of each row of this kind of message, the signal itself
                        if series.size > 0:
                                # store signal as message id + its number, something like: 0110_1, first signal of message 0110
                                df[f"{id}_{index}"] = np.interp(t, series.index.values, series.values)#, left=np.NaN, right=np.NaN)

        

        # ------------------------ Additional preprocessing ------------- 
        #fill missing values with 0 and set timestamp as index
        df_without_const = df.fillna(0)
        #print(df_without_const.index)#df_without_const = df_without_const.set_index(df.index[:-1])

        #keep only interesting_columns
        columns_to_keep = list(interesting_columns.intersection(df_without_const.columns))
        #notify if not all columns found in data
        if len(columns_to_keep) != len(interesting_columns):
                log("Preprocess warning", "Some data not used, some columns specified in interesting_columns not found in data. \nColumns not in final columns (but specified earlier): " + str([value for value in interesting_columns if value not in columns_to_keep]) + "\nOriginal df columns: " + str(df_without_const.columns) + "\n Final columns: " + str(columns_to_keep))
                raise ValueError("Not all signals found.")
        #df_without_const = df_without_const[columns_to_keep] # NOTE: comment-out if only relevant signals are needed

        

        # ------------------------ Scale each signal  ------------- 
        # define a simple scaling function, which will scale the data with a given max value into 0-1 range
        scale = np.frompyfunc(lambda x, min, max: (x - min) / (max - min), 3, 1)

        # Scale each signal to 0-max values
        df_scaled = pd.DataFrame(columns= df_without_const.columns)
        
        # iterate over all signals
        for col in df_scaled.columns:
                message_id = col.split('_')[0]
                signal_index = int(col.split('_')[1])
                # get all rows in signal mask with this id and iterate over these rows (these will be signals with the same message id)
                rows = signal_mask.query(f'id=="{message_id}"')
                row = rows[rows['index']==signal_index]

                
                # index of the singal in every singal of the corresponding message (first signal, second, etc.)
                index = int(row['index'].iloc[0])
                
                if index != signal_index:
                        raise ValueError("something wrong, indexes do not match: " + str(index) + ", " + str(signal_index) + ", column: " + str(col))
                
                # start and stop index of bits in the message (first signal is for example from 7. bit to 16. bit)
                start = int(row['start'].iloc[0])
                stop = int(row['stop'].iloc[0])
                # we want ot scale with the maximum value of the signal, which can be stored in that many bits
                num_bits = stop-start
                max_value = 2 ** num_bits
                # name of the signal, consists of the message and the signal's id, like 0110_1
                signal = str(message_id)+'_'+str(int(index))

                # Scale signal with its scaler
                column = df_without_const[signal] # get the column of singal
                x = np.array(column).reshape(-1,1) # reshape for scaling
                y = scale(x, 0, max_value) # scale
                if minmaxScale: 
                        scaler = MinMaxScaler()
                        y = scaler.fit_transform(x)
                df_scaled[signal]= list(y[:,0]) # reshape again and store store df_scaled

        #log("... preprocessing done!")
        if df_scaled.isnull().values.any():
                log("Preprocess value error", "NaN values found in scaled dataframe (should not be, filled with zeros)")
                raise ValueError("NaN values found in scaled dataframe")
        return df_scaled

def should_preprocess(filename: str, debug_info = False) -> bool:

        allowed_filename_parts = ['msg-mod', 'benign', 'trace', 'msg-inj']
        

        if filename is not None and filename.endswith(('can.csv')):
                log("Log file provided, it only has csv-specific preprocessing, further preprocessing data...")
                return True
        
        elif "SynCAN" in filename:
                if "train_input" in filename:
                        if debug_info:
                                log("\nSynCAN dataset, simply reading interpolated train traces from csv...")
                        return False
                if "unprocessed_csv_files" in filename:
                        if debug_info:
                                log("\nSynCAN dataset, reading unprocessed csv files...")
                        return True
                
        elif filename is not  None and any(a in filename for a in allowed_filename_parts) and filename.endswith(('.log')):
                log("Log file provided, preprocessing data ...")
                return True
        
        elif filename is not  None and filename.endswith('.csv'):
                raise Exception("This file is probably already scaled with a previous scaler. Please check, and use the new scaler. (We do not store preprocessed (and scaled) data now...)")

        elif filename is not  None and filename.endswith('.txt'):
                raise Exception("This is a preprocessed file, just reading")
        
        else:
                log("Got wrong filename: " + filename, emphasize=True)
                raise Exception("No test path was provided")


def load_data(filename: str, config: Config, signal_mask: None = None, debug_info: Optional[bool] = None) -> Tuple[DataFrame, List[str], int]:
        
        preprocess_needed = should_preprocess(filename)

        


        # ---------------  Get information from config ------------
        #num_signals = sum([len(signal_ids_dict[group_number]["signals"]) for group_number in signal_ids_dict])
        ID_groups: List[List[str]] = config.signal_groups

        if signal_mask is None:
                
                if 'trace' in filename:
                        signal_mask = signal_mask_path_old_traces
                        log("Using modified new signal mask for old traces: " + signal_mask_path_old_traces)
                else:
                        signal_mask = signal_mask_path
                if debug_info:
                        log("Using scaler from config.py: " + signal_mask)
        elif 'old' in signal_mask:
                signals = flatten([ID_groups[item] for item in ID_groups])
                
                if '0410_1' not in signals:
                        log("Changing column names back to old ones, due to old signal mask")
                        from scripts.config import column_dict
                        for g, group in enumerate(ID_groups):
                                new_group = []
                                for new in group:
                                        new_group.append(column_dict[new])
                                ID_groups[g] = new_group
                        raise Warning("Check if new signal names are correct, using ID_groups as list instead of dictionary")


        # -------------- Get base test dataframe ------------------
        # it will contain all signals, scaled, ready to use
        if preprocess_needed:
                # load log file, interpret signal values, scale, and return in a dataframe
                df_test = df_from_log(filename, path_signal_mask = signal_mask)
                if debug_info:
                        log("Loaded dataframe from log file", "Columns: " + str(df_test.columns) + "\nHead: \n" + str(df_test.head()))
        else:
                # Read test file from csv - if stored in csv, all preprocess done
                df_test = pd.read_csv(filename)


        

        # --------------- Reshape dataframe according to grouping -
        # df_test_rearranged - test df to use in prediction, same group signals are next to each other
        # columns_to_keep - signal ID-s in the appropriate order
        rearranged_columns = []
        cols = 0
        # signal ID-s in config are always in an array, if we only got 1 signal, for loop would return characters in its ID, so we treat it differently
        if len(ID_groups) != 1:
                
                # iterate each group
                for group in ID_groups:
                        # iterate each signal in group
                        for index in group:
                                rearranged_columns.append(index)

                print(rearranged_columns)
                print(df_test.columns)
                # make new df with the rearranged columns
                df_test_rearranged = df_test[rearranged_columns]
                
                # cols will tell how many signals to visualize in a row when calling visualize_signals, not actual column number
                cols = 3
                
        #if one signal only
        else:
                df_test_rearranged = df_test[ID_groups[0]]
                cols = 1
                rearranged_columns = ID_groups[0]
               
        # print some info
        if debug_info:
                log("df_test_rearranged" , str(df_test_rearranged.head()))
                log("Loaded this dataframe:")
                visualize_signals(df_test_rearranged)

        return df_test_rearranged, rearranged_columns, cols





# Load data from log file
# interpret signals, scale, return in dataframe
def df_from_log_part(log_file: str, read_from: Optional[int] =None, nrows: Optional[int]=None, path_signal_mask: Optional[str]=None, minmaxScale: bool=False, raw_data=False, time_as_index = True) -> DataFrame:

        # if only one is None from read_from and nrows
        if (read_from is None) != (nrows is None):
                raise ValueError("Either both read_from and nrows should be None (meaning the whole file), or both should be specified")

        # ------------------------ Setting up signal mask -------------

        if path_signal_mask is None:
                if 'trace' in log_file:
                        path_signal_mask = signal_mask_path_old_traces
                        log("Using modified new signal mask for old traces: " + signal_mask_path_old_traces)
                else:
                        path_signal_mask = signal_mask_path
                        

        elif 'old' in path_signal_mask:
                log("Changing interesting column names back to old ones, due to old signal mask")
                from scripts.config import old_columns
                interesting_columns = old_columns
        else:
                interesting_columns = interesting_columns_


        # --------------------------------------------------------------- #
        # ------------------------ Load dataframe from file ------------- #
        # --------------------------------------------------------------- #
        
        
        # ------------------------ Loading parameters ----------------------
        # -------------- determining how much more data should be loaded than needed, so we can interpolate the whole part
        
        original_read_from = read_from
        original_nrows = nrows
        if read_from is not None:
                log(f"Reading part of file, from {read_from} to {read_from+nrows}")
        
                first_index = read_from
                last_nrows = nrows
                for id in interesting_columns:

                        # getting message id part of the signal id
                        message_id = id.split('_')[0]

                        last_timestamp_before = get_index_of_next_or_previous_message(log_file = log_file, read_from=read_from, nrows=nrows, message_id=message_id, before_timestamp=True)
                        first_timestamp_after = get_index_of_next_or_previous_message(log_file = log_file, read_from=read_from, nrows=nrows, message_id=message_id, before_timestamp=False)

                        first_index = min(first_index, last_timestamp_before)
                        last_nrows = max(last_nrows, first_timestamp_after)

                read_from = first_index
                nrows = last_nrows

        assert not (first_index != read_from and last_nrows != nrows) # only changing one or none of them


        if "SynCAN" in log_file: 
                # Load the data
                df_test = pd.read_csv(log_file, skiprows=read_from, nrows=nrows)
        else:
                # -------------- loading
                # reading part of log file
                df_test = pd.read_csv(log_file, skiprows=read_from, nrows=nrows, sep=' \s+', header=None, engine='python', dtype=str)

        if df_test.empty:
                with open(f"{work_dir}/_logs/error_outputs/ERROR_LOGS_load_data.txt", 'w') as f:
                        f.write("-------------------------------- error during loading data ------------------------------------\n")
                        f.write("(this file is overwritten every time, this is the last error)")
                        f.write(f"read_from: {read_from}\n")
                        f.write(f"nrows: {nrows}\n")
                        f.write(f"log file is {log_file}\n")
                        f.write(f"df test is {df_test}\n")
                raise Exception("Empty dataframe, logged error in error_outputs/ERROR_LOGS_load_data.txt")

        if "SynCAN" in log_file:
                df_temp = df_test
                # interpolate the data
                df_test = pd.DataFrame()
                
                t = df_temp.index 
                for id in df_temp["ID"].unique():

                        # Filter on message ID
                        filtered = df_temp.query(f'ID=="{id}"')

                        # If there are less than 3 messages, we can't interpolate
                        if len(filtered) < 3:
                                raise Exception("Not enough message to interpolate (probably)")
            
                        # iterate over the 4 signals, and interpolate them, then add them to the final df
                        for i in range(1,5):
                                # get all not nan rows
                                not_nan = filtered[filtered[f"Signal{i}_of_ID"].notna()]
                
                                # if no not nan rows, break
                                if len(not_nan) == 0:
                                        break

                                df_test[f"{id}_{i}"] = np.interp(t, not_nan.index, not_nan[f"Signal{i}_of_ID"])
                
                # rearrange columns will be done on load_data function after returning

                return df_test

        # Add column names
        df_test.columns = ['TimeStamp', 'ID0', 'ID1', 'DLC_length', 'DLC_bytes']

        # Convert timestamps to float start from 0 and set as index
        df_test['TimeStamp'] = df_test['TimeStamp'].astype(float)
        df_test['TimeStamp'] -= df_test['TimeStamp'][0]
        df_test = df_test.set_index('TimeStamp')

        # Convert DLC_bytes to BitArray
        df_test['DLC_bytes'] = df_test['DLC_bytes'].map(lambda s: BitArray(hex=s))

        # ------------------------ Extract signals from log data ------------- 
        # read saved signal mask from file
        store = pd.HDFStore(path_signal_mask)
        signal_mask = store['signal_mask']
        store.close()

        # We will use the signal_mask, so make sure it is set properly and exists
        if signal_mask is None:
                raise ValueError("No signal mask found, because either path_signal_mask is not specified in data_preprocess.py, or no signal mask exists. Use calculate_signals/src/signal_extractor.py and then calculate_signals/signal_mask.py to get a signal mask") 

        
        #keep only interesting_columns
        columns_to_keep = interesting_columns
        
        # extract signals from can logs with signal mask
        t = df_test.index 
        df = pd.DataFrame() # this df will contain all signals from the logs in separate columns

        # iterate over only interesting columns
        for signal in columns_to_keep:
                # message id
                id = signal.split('_')[0]

                # Filter on message ID
                filtered: DataFrame = df_test.query(f'ID0=="{id}"')
                if filtered.shape[0] < 3:
                        with open(f"{work_dir}/_logs/error_outputs/load_data_output.txt", 'w') as f:
                                f.write("-------------------------------- error during filtering messages ------------------------------------\n")
                                f.write("(this file is overwritten every time, this is the last error)")
                                f.write(f"filtered on id {id}\n")
                                f.write(filtered.to_string()+"\n")
                                f.write(str(read_from)+"\n")
                                f.write(str(nrows)+"\n")
                                f.write("\n")
                        raise Exception(f"Not enough message to interpolate (probably), length of filtered dataframe: {len(filtered)}")
                
                # Extract time series using slices
                # iterate over each message id, they will contain multiple signals
                for _, row in signal_mask.query(f'id=="{id}"').iterrows():
                        index = int(row['index'])
                        if index < 0:
                                raise ValueError('Index is ' + str(index) + ", shouldn't be below 0")
                        start = row['start'] # start position of current signal
                        stop = row['stop'] # end position of it
                        part = slice(int(start), int(stop)) 
                        series: DataFrame = filtered['DLC_bytes'].map(lambda x: x[part].uint) # get this part of each row of this kind of message, the signal itself
                        if series.size > 0:
                                # store signal as message id + its number, something like: 0110_1, first signal of message 0110
                                df[f"{id}_{index}"] = np.interp(t, series.index.values, series.values)#, left=np.NaN, right=np.NaN)
                                
                                interpolated_zeros_original_not_zeros = True
                                for value in df[f"{id}_{index}"]:
                                        if value != 0:
                                                interpolated_zeros_original_not_zeros = False
                                                break
                                for value in series.values:
                                        if value == 0:
                                                interpolated_zeros_original_not_zeros = False
                                                break
                                if interpolated_zeros_original_not_zeros:
                                        with open(f"{work_dir}/_logs/error_outputs/ERROR_LOGS_load_data.txt", 'a') as f:
                                                f.write("--------------------------------------------------------------------\n")
                                                f.write(f"Set from filtered series: {set(series.values)}\n")
                                                f.write("---------- Filtered part: ----------\n")
                                                f.write(str(series.values))
                                                f.write("\n")
                                                f.write("Interpolated part as set (types of values):\n")
                                                f.write("\t" + str(set(df[f"{id}_{index}"].values)))
                                                f.write("\n")
                                                f.write(f"id is {id}\n")
                                                f.write(f"index is {index}\n")
                                        raise Warning("Interpolated zeros")
                        else:
                                log(f"""No message in this part
                                        \t\tfrom {read_from} to plus {nrows}
                                        \t\tstart {start}, stop {stop}
                                        \t\tmessage id {id}
                                        \t\tindex {index}\n""")
                                raise Exception("No message part found")


        # ------------------------ Get part of dataframe that was originally needed -------------

        df = df.iloc[original_read_from-first_index : original_read_from-first_index + original_nrows]

        # ------------------------ Additional preprocessing ------------- 
        #fill missing values with 0 and set timestamp as index
        df_without_const = df.fillna(0)
        #print(df_without_const.index)#df_without_const = df_without_const.set_index(df.index[:-1])

        # ------------------------ Scale each signal  ------------- 
        # define a simple scaling function, which will scale the data with a given max value into 0-1 range
        scale = np.frompyfunc(lambda x, min, max: (x - min) / (max - min), 3, 1)

        # Scale each signal to 0-max values
        df_scaled = pd.DataFrame(columns= df_without_const.columns)
        
        # iterate over all signals
        for col in df_scaled.columns:
                message_id = col.split('_')[0]
                signal_index = int(col.split('_')[1])
                # get all rows in signal mask with this id and iterate over these rows (these will be signals with the same message id)
                rows = signal_mask.query(f'id=="{message_id}"')
                row = rows[rows['index']==signal_index]

                
                # index of the singal in every singal of the corresponding message (first signal, second, etc.)
                index = row['index'].iloc[0]
                
                if index != signal_index:
                        raise ValueError("something wrong, indexes do not match: " + str(index) + ", " + str(signal_index) + ", column: " + str(col))
                
                # start and stop index of bits in the message (first signal is for example from 7. bit to 16. bit)
                start = int(row['start'].iloc[0])
                stop = int(row['stop'].iloc[0])
                # we want ot scale with the maximum value of the signal, which can be stored in that many bits
                num_bits = stop-start
                max_value = 2 ** num_bits
                # name of the signal, consists of the message and the signal's id, like 0110_1
                signal = str(message_id)+'_'+str(int(index))

                # Scale signal with its scaler
                column = df_without_const[signal] # get the column of singal
                x = np.array(column).reshape(-1,1) # reshape for scaling
                y = scale(x, 0, max_value) # scale
                if minmaxScale: 
                        scaler = MinMaxScaler()
                        y = scaler.fit_transform(x)
                df_scaled[signal]= list(y[:,0]) # reshape again and store store df_scaled

        #log("... preprocessing done!")
        return df_scaled

def load_data_part(filename: str, read_from:int, read_to:int, config: Config, signal_mask: None = None, debug_info: Optional[bool] = None) -> Tuple[DataFrame, List[str], int]:


        preprocess_needed = should_preprocess(filename)

        
        # ---------------  Get information from config ------------
        #num_signals = sum([len(signal_ids_dict[group_number]["signals"]) for group_number in signal_ids_dict])
        ID_groups: List[List[str]] = config.signal_groups

        if signal_mask is None:
                
                signal_mask = signal_mask_path
                if debug_info:
                        log("Using scaler from config.py: " + signal_mask_path)

        elif 'old' in signal_mask:
                signals = flatten([ID_groups[item] for item in ID_groups])
                
                if '0410_1' not in signals:
                        log("Changing column names back to old ones, due to old signal mask")
                        from scripts.config import column_dict
                        for g, group in enumerate(ID_groups):
                                new_group = []
                                for new in group:
                                        new_group.append(column_dict[new])
                                ID_groups[g] = new_group
                        raise Warning("Check if new signal names are correct, using ID_groups as list instead of dictionary")


                # -------------- Get base test dataframe ------------------
        # it will contain all signals, scaled, ready to use
        if preprocess_needed:
                 # load log file, interpret signal values, scale, and return in a dataframe
                df_test = df_from_log_part(log_file = filename, read_from = read_from, nrows=read_to, path_signal_mask = signal_mask)
                if debug_info:
                        log("Loaded dataframe from log file", "Columns: " + str(df_test.columns) + "\nHead: \n" + str(df_test.head()))
        else:
                if "SynCAN" in filename:
                        if read_from != 0:
                                # Read part of test file from csv - if stored in csv, all preprocess done
                                df_test = pd.read_csv(filename, skiprows=read_from, nrows=read_to, sep=',',  dtype=float, header=None)
                                # set column names according to first row in file
                                #with open(filename, 'r') as f:
                                #        column_names = f.readline().strip().split(',')
                                df_test.columns = ['id5_1', 'id5_2', 'id8_1', 'id3_1', 'id3_2', 'id7_1', 'id7_2', 'id9_1','id1_1', 'id1_2', 'id2_1', 'id2_2', 'id2_3', 'id6_1', 'id6_2', 'id10_1','id10_2', 'id10_3', 'id10_4', 'id4_1']
                        else:
                                df_test = pd.read_csv(filename, skiprows=read_from, nrows=read_to, sep=',',  dtype=float)
                                
                        if debug_info:
                                log("Loaded part of dataframe from csv file", "Columns: " + str(df_test.columns) + "\nHead: \n" + str(df_test.head()))
                else:
                        raise Exception("Currently not supporting other preprocessed files than SynCAN dataset, please implement it if needed (for example, csv separators could be different, etc.)")


        # --------------- Reshape dataframe according to grouping -
        # df_test_rearranged - test df to use in prediction, same group signals are next to each other
        # columns_to_keep - signal ID-s in the appropriate order
        rearranged_columns = []
        cols = 0
        # signal ID-s in config are always in an array, if we only got 1 signal, for loop would return characters in its ID, so we treat it differently
        if len(ID_groups) != 1:
                
                # iterate each group
                for group in ID_groups:
                        # iterate each signal in group
                        for index in group:
                                rearranged_columns.append(index)

                # make new df with the rearranged columns
                df_test_rearranged = df_test[rearranged_columns]
                
                # cols will tell how many signals to visualize in a row when calling visualize_signals, not actual column number
                cols = 3
                
        #if one signal only
        else:
                df_test_rearranged = df_test[ID_groups[0]]
                cols = 1
                rearranged_columns = ID_groups[0]
               
        # print some info
        if debug_info:
                log("df_test_rearranged" , str(df_test_rearranged.head()))
                log("Loaded this dataframe:")
                visualize_signals(df_test_rearranged)

        return df_test_rearranged.values

# getting last / first message before / after a given timestamp (given with index in lines of the file) and message id
# --- if before_timestamp is True: returns index of last message before timestamp, to be used as a read_from value in a read_csv function
# --- if before_timestamp is False: returns relative index of first message after timestamp, to be used as a nrows value in a read_csv function
# returns index of the message in the dataframe (or relative index of the message if before_timestamp is False)
def get_index_of_next_or_previous_message(log_file: str, read_from:int, nrows:int, message_id: str, before_timestamp: bool) -> DataFrame:

        # handle if read_from is 0
        if read_from == 0:
                return read_from

        # how many lines to read more than needed
        plus_part = 100

        # determine which way to load more data (before / after)
        read_from_ = read_from - plus_part if before_timestamp else read_from
        nrows_ = nrows + plus_part 

        if dump_debug_log_to_file:
                with open(f"{work_dir}/_logs/log_outputs/LOG_get_next_prev_index.txt", "w") as f:
                        f.write("--------------------------------log for get_index_of_next_or_previous_message function------------------------------------\n")
                        f.write("(only logged if dump_debug_log_to_file is True)")
                        f.write("plus part is " + str(plus_part))
                        f.write("\n")
                        f.write("read_from is " + str(read_from))
                        f.write("\n")
                        f.write("nrows is " + str(nrows))
                        f.write("\n")
                        f.write("read_from_ is " + str(read_from_))
                        f.write("\n")
                        f.write("nrows_ is " + str(nrows_))
                        f.write("\n")
                        f.write("before_timestamp is " + str(before_timestamp))
                        f.write("\n")
                        f.write("message_id is " + str(message_id))
                        f.write("\n")

        # reading part of log file
        df_test = pd.read_csv(log_file, skiprows=read_from_, nrows=nrows_, sep=' \s+', header=None, engine='python', dtype=str)

        # Add column names
        df_test.columns = ['TimeStamp', 'ID0', 'ID1', 'DLC_length', 'DLC_bytes']

        # Convert timestamps to float start from 0 and set as index
        df_test['TimeStamp'] = df_test['TimeStamp'].astype(float)
        df_test['TimeStamp'] -= df_test['TimeStamp'][0]
        #df_test = df_test.set_index('TimeStamp')

        # Convert DLC_bytes to BitArray
        df_test['DLC_bytes'] = df_test['DLC_bytes'].map(lambda s: BitArray(hex=s))

        # get all messages with this id
        messages: DataFrame = df_test.query(f'ID0=="{message_id}"')

        if dump_debug_log_to_file:
                with open(f"{work_dir}/_logs/log_outputs/LOG_get_next_prev_index.txt", "a") as f:
                        f.write("--------- messages --------\n" + str(messages.to_string()))
                        f.write("\n")

        # determine timestamp
        timestamp = df_test["TimeStamp"].iloc[plus_part]

        if before_timestamp:
                # get last message before timestamp
                last_message = messages[messages["TimeStamp"] <= timestamp].tail(1)
        else:
                # get first message after timestamp
                last_message = messages[messages["TimeStamp"] >= timestamp].head(1)

        if dump_debug_log_to_file:
                with open(f"{work_dir}/_logs/log_outputs/LOG_get_next_prev_index.txt", "a") as f:
                        f.write("--------- last message ----\n" + str(last_message))
                        f.write("\n")
                        f.write("last messsage timestamps is " + str(last_message['TimeStamp']))
                        f.write("\n")
                        f.write("timestamp is " + str(timestamp))
                        f.write("\n")
                        f.write("df_test is " + str(df_test.to_string()))
                        f.write("\n")



        # get relative index of this message (get row where timestamp is the same as the last message's timestamp)
        index_ = messages.index[messages['TimeStamp'] == timestamp]

        # get absolute index of this message (relative index + read_from)
        if before_timestamp:
                index = index_ + read_from_
        else:
                # in this case, it won't be an abosulte index, but will be used as "how many rows to read from read_from", so relative needed anyway
                index =  index_

        return index
