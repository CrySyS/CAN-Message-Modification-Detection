import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from modules.scripts.visualization import visualize_signals, log
from modules.scripts.data_preprocess import load_data, df_from_log
from modules.scripts.config import TrainedModels
from modules.scripts.utils import get_group_index_by_signal_name
from functools import reduce
from modules.scripts.datagenerator import MultipleFileGenerator
import json
import os
from modules.scripts.config import input_data


# -------------------------------- USED FUNCTIONS --------------------------------

######################################################################################
# ---------------- PREDICTION -------- 
# ------------------------------------
# used in: evaluate.py, threshold.py
# this function is used to make prediction with a model for a test file

# predict with model for a test file, and plot with visualize_signals from visualization.py
def multistep_predict(model, config, filename, debug_info = False, plot = False, signal_mask = None, save = False):

        if signal_mask is None:
                from modules.scripts.config import signal_mask_path
                signal_mask = signal_mask_path
                if debug_info:
                        log("Using stored signal mask: " + signal_mask_path)

        # print some info
        if debug_info:
                log("Multistep evaluate start -----------------------------------------------------------------------\n\n", filename)

        # ---------------  Get information from config ------------
        signal_groups = config.signal_groups
        num_signals = sum([len(group) for group in signal_groups]) # number of signals, regardless of grouping
        ts = config.get("window_size") # time steps, size of sliding window
        target_values = config.get("multistep") # how many values to forecast

        # --------------- Reshape dataframe according to grouping -
        #print("signal mask:", signal_mask)
        df_test_rearranged, rearranged_columns, cols = load_data(filename=filename, config=config, signal_mask=signal_mask, debug_info=debug_info)
        
        #if debug_info:
        #        print(f"loaded these rearranged columns: {rearranged_columns}")
        #        print(f"loaded dataframe head is: {df_test_rearranged.head}")

        # ----------------- Sliding windows and prediction ---------
        # get values from test dataframe
        X_test = df_test_rearranged.values
        
        # Create windowed data from test df 
        #test_samples_generator = singlestep_datagenerator(X = X_test, time_steps=ts, sampling_rate=1, stride=1)
        #test_samples_generator = multistep_datagenerator(X_test, sample_length=ts, target_length=target_values, only_x=True) # for prediction
        test_samples_generator = MultipleFileGenerator(file_paths=[filename], window_size=config.window_size, batch_size=config.batch_size, config=config, part=(0,1), debug_info=False)
        #test_samples_generator = SingleFileGenerator(file_path=filename, confnumber=1, only_input=True, debug_info=debug_info)

        # Make prediction with the model
        # predict with model in this scope, so GPU will use all devices
        # set tensorflow to use all available GPU devices
        strategy = tf.distribute.MirroredStrategy(devices=None)
        with strategy.scope():
                X_predict = model.predict(test_samples_generator)


        # print some info
        if debug_info:
                log("Data shapes: ", f"X predict: {X_predict.shape} \ndf test: {df_test_rearranged.shape} \nX test: {np.array(X_test).shape} \ngenerator: {len(test_samples_generator)}")
                

        # -----------------------  Format final prediction ----------------------------
        # Create final prediction from what we got from model
        #X_predict_original = np.zeros(shape=X_test.shape) # creating a testdata-big array with zeros to fill in with the predictions

        #discard batches - X_predict shape: (batch, target values, num signals) -> we want (all rows, num signals)
        X_predict_original = np.reshape(X_predict, (-1, num_signals))

        # we don't want to pad with zeros, because 
        # adding padding at the beginning of each group, padding value is always the first value
        assert len(X_predict_original[0]) == num_signals
        padding = pd.DataFrame(data = [X_predict_original[0]] * ts, columns=rearranged_columns)
        predicted_signals_with_padding = np.concatenate([padding, X_predict_original])

        #adding padding, zeros at the beginning (for first sliding window)
        #padding = pd.DataFrame(data = np.zeros((ts, num_signals)), columns=rearranged_columns)
        #X_predict_original = np.concatenate([padding, X_predict_original])
        
        #adding padding at the end
        assert len(X_test) == len(predicted_signals_with_padding), f"length of test data ({len(X_test)}) and prediction ({len(predicted_signals_with_padding)}) do not match (should!)"
        #length_padding_end = len(X_test) - len(X_predict_original)
        #padding_end = pd.DataFrame(data = np.zeros((length_padding_end, num_signals)), columns=rearranged_columns)
        #X_predict_original = np.concatenate([ X_predict_original, padding_end])


        # -----------------------  Plot test & prediction ------------------------------
        #plot test and predicted
        df_pred = pd.DataFrame(data=predicted_signals_with_padding, columns = df_test_rearranged.columns)
        if plot: visualize_signals(df_test_rearranged, df_pred, filename=filename, columns=cols, num_signals=num_signals, save = save)


        # ----------------------- Plot loss --------------------------------------------
        # plot loss over data
        if len(X_test) == len(predicted_signals_with_padding):
                loss = tf.keras.losses.mse(X_test, predicted_signals_with_padding)
                
                #print("Threshold Loss shape: " + str(loss.shape))
                stuff = np.percentile(loss[0], 0.99)#TODO: later

        else:
                raise Exception("length of test data and prediction do not match (should!)")


        # print some info
        if debug_info:
                log("Multistep evaluate end -----------------------------------------------------------------------", emphasize=True)

        #return prediction for later use (save)
        return predicted_signals_with_padding, df_test_rearranged, df_pred



######################################################################################
# -------------- PRED -> LOSS --------
# ------------------------------------
# used in: evaluate.py, threshold.py
# these functions are used to calculate losses, and the sum / median in evaluation window for a test file
#

#Calculates loss for each GROUP and plots them
def calculate_group_loss(test_df, pred_df, config, plot = False):
        

        X_test = test_df.values
        X_pred = pred_df.values

        groups = config.signal_groups

        group_losses = []
        if(len(groups) != 1): #will fail when trying to iterate less than 20 signals WITHOUT groups, but that's ok, we don't use that
                # iterate over each group
                for group in groups:
                        signals = test_df[group] # test signals in group
                        pred = pred_df[group] # predicted signals in group
                        #log(f"Shape of signals: {signals.shape}")
                        #log(f"Shape of pred: {pred.shape}")
                        #log(f"MSE loss of signals and pred: {tf.keras.losses.mse(signals, pred).numpy()}")

                        group_losses.append(tf.keras.losses.mse(signals, pred).numpy()) # NOTE: a .numpy() azért kellett mert egy elemű tömböket pakolt a loss értékek helyett egy tömbe valamiért enélkül (egy komplexebb objektumot adott vissza, nekünk csak a tömb kell)
                        #visualize_signals(signals, pred, num_signals=len(group))

        else:
                log("Note when calculating losses of each signal group: One group or only one signal")
                group_losses.append(tf.keras.losses.mse(X_test, X_pred).numpy()) # NOTE: lásd fentebb pár sorral

        
        if plot:
                for index in groups:
                        i = index -1
                        var = groups[index]
                        plt.figure(figsize=(18, 6))
                        plt.plot(np.arange(len(group_losses[i])), np.array(group_losses[i]), 'r', label=f"Loss {var}")
                        
                        
                        for j in range(len(var)):
                                alpha_mod = 1 / len(var) * (j+1)
                                plt.plot(np.arange(len(group_losses[i])), np.array(test_df[var])[:,j], 'g', markersize=2, alpha=0.6*alpha_mod,
                                        label=f"Signal {var}")
                                plt.plot(np.arange(len(group_losses[i])), np.array(pred_df[var])[:,j], 'b', markersize=2, alpha=0.6 * alpha_mod,
                                        label=f"Predicted signal {var}")

                        plt.legend(loc='upper left')
                        plt.show()
        
        # Prediction starts after the first window, so those first loss values will be bad, we replace them with zeros
        window_size = config.get('window_size')
        group_losses = np.array(
                        [np.append(  np.zeros(window_size),  sublist[window_size:] ) for sublist in group_losses]
                        )
        # also the last window size values
        group_losses = np.array(
                [np.append(  sublist[ : -window_size], np.zeros(window_size) ) for sublist in group_losses]
                )
        
        return group_losses

# evaluation window modes --------
# A) sum loss values in evaluation window
def sum_window(loss_window):
       return reduce(lambda a, b: a+b, loss_window)

# B) calculate median of loss values in evaluation window
def median_window(loss_window):
       loss_window.sort()
       return loss_window[len(loss_window)//2]


# calculate sum / median of last window size losses for each time step
def calculate_summed_losses(group_losses, window_size):
        assert window_size > 0, "Please specify a valid window size for evaluation"

        # this array will store the summed losses for each time step for each group (List of lists)
        areas_grouped: list[list[float]] = []
        lenght_of_data = len(group_losses[0])
        # iterate over each group of losses
        for group_loss in group_losses:
                # the first window_size values will be 0, because we can't calculate the sum of the previous window_size values for them
                areas = [0]* (window_size)
                # iterate over each remaining time step
                for ndx in range(lenght_of_data-window_size):
                        # get the window_size values after the current time step
                        loss_window = group_loss[ndx: ndx + window_size]

                        assert len(loss_window) == window_size
                        # calculate the sum of the losses in the window
                        area = sum_window(loss_window)
                        #median = median_window(loss_window)  #NOTE: this is not an area now
                        # append the sum to the list of the current summed group loss
                        areas.append(area)
                # append the list of the group to the list of all summed losses
                areas_grouped.append(areas)
        return areas_grouped
                               
              
######################################################################################
# --------- LOSS -> DETECTION --------
# ------------------------------------
# used in: evaluate.py, threshold.py
# these functions are used to detect sum/median loss values that are higher than a set threshold in a test file with reading the true anomalies from the log


# detect with evaluation windows and thresholds
def area_detection(areas_grouped, model_id, entire_array = False):
        # Read threshold values from saved config
        tm = TrainedModels()
        model_config = tm.get_data(model_id)
        thresholds = model_config['area_thresholds']

        thr_detection_indexes_grouped = [] # indexes where value of loss is higher than threshold
        #thr_detection_percentage_grouped = {} # loss/threshold percentage and indexes
        all_array = []
        binary_array = []
        for i,loss in enumerate(areas_grouped):
            threshold = float(thresholds[i])

            temp_indexes = []
            temp_array = []
            temp_binary = []

            for j in range(len(loss)):
                if loss[j] >= threshold:
                    
                    value = loss[j]/threshold

                    temp_indexes.append(j)
                    temp_array.append(value)
                    temp_binary.append(1)
                else:
                    temp_array.append(0)
                    temp_binary.append(0)

            thr_detection_indexes_grouped.append(temp_indexes)
            all_array.append(temp_array)
            binary_array.append(temp_binary)

        if entire_array:
                return thr_detection_indexes_grouped, all_array, binary_array
        return thr_detection_indexes_grouped

# returns a full length array, with 0s where no anomalies, and 1s where true anomalies
def read_true_anomalies(test_file):



        # read json file instead
        df_log = df_from_log(log_file=test_file, raw_data=True, time_as_index=False)

        new_format = 'can' in df_log.columns # we have a separate column for attack flag
        if new_format:
                file_name = os.path.basename(test_file)
                json_file = f"{input_data}/log_meta_data/{file_name[:-4]}.json"
                with open(json_file) as f:
                        json_data = json.load(f)
                
                markers = json_data['markers']
                first_flag_time = markers[0]['time']
                last_flag_time = markers[1]['time']

        y_true = []

        for i in df_log.index: # we should iterate the index ( not timestamp), because in case of an injection attack, the timestamp is not unique

                if new_format:
                        # if timestamp is between the first and last flag, it is an attack
                        if first_flag_time <= df_log['TimeStamp'][i] <= last_flag_time:
                                y_true.append(1)
                        else:
                                y_true.append(0)

                
                                
                                '''log(f"Dataframe head: {df_log.head()}")

                                if df_log['attacked_flag'][i] == 'True':
                                        y_true.append(1)
                                else:
                                        y_true.append(0)'''

                # old log format
                else:
                        txt = str(df_log['DLC_bytes'][i])
                        try:

                                y_true.append(int(txt[-1]))
                        except:
                                log("Error in reading true anomalies from log file", emphasize=True)

                                log("Reading true anomalies from log file", emphasize=True)
                                # log the value being read
                                log("Value being read as int:\n" + txt)
                                # log the index of the value being read
                                log("Indexed value being read as int:\n" + txt[-1])

                                log("Index of value being read: " + str(i))

                                log(df_log['DLC_bytes'][i])

                                raise Exception("Error in reading true anomalies from log file")

        if 1 in y_true:
                first_occ = y_true.index(1)
                last_occ = len(y_true) - 1 - y_true[::-1].index(1)

                for i in range(first_occ, last_occ+1):
                        y_true[i] = 1
        
        return y_true


######################################################################################
# -------- DETECTION -> METRICS --------
# --------------------------------------
# used in: evaluate.py, threshold.py
# these functions are used to calculate metrics for a test file with the detection results

# calculate all metrics (fpr, detections, times to detect) for a test file
def calculate_metrics(test_file, model_config, binary_array, evaluation_window_size, output_folder, debug_info = False, save = True, benign = False):

        # -----------------------  True anomalies --------------------------------------------------
        # Read original test file to collect true attack points (binary)
        if "benign" in test_file:
                true_anomalies = np.zeros(len(binary_array[0]))
        else:
                true_anomalies = read_true_anomalies(test_file)
        

        # -----------------------  Attacked group -------------------------------------------------
        attacked_groups = []
        if not benign:
                # Determine which signal is attacks according to the test file name
                attack_in_signal = test_file[test_file.find('0x'):]
                attack_in_signal = attack_in_signal[:attack_in_signal.find('-')]
                attack_in_signal = attack_in_signal.replace('x', '') # attack_in_signal will be like '0410'

                
                # NOTE: this has to be generalised if tested with another dataset
                if '0410' in attack_in_signal:

                        attack_group = get_group_index_by_signal_name('0410_0', model_config)
                        attacked_groups.append(attack_group)
                        attacked_signals = binary_array[attack_group]

                elif '0110' in attack_in_signal:
                        
                        attack_group = get_group_index_by_signal_name('0410_0', model_config)
                        attacked_groups.append(attack_group)
                        attacked_group_1 = binary_array[attack_group]

                        attack_group = get_group_index_by_signal_name('0110_0', model_config)
                        attacked_groups.append(attack_group)
                        attacked_group_2 = binary_array[attack_group]

                        # Merge the arrays using logical OR
                        attacked_signals = np.logical_or(attacked_group_1, attacked_group_2).astype(int)
                        attacked_signals = attacked_signals.tolist()
                else:
                        raise Warning("Attacked signal name not as expected, probably evaluating other attacks.")


        # -----------------------  Metrics --------------------------------------------------------
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # read raw data to get timestamps
        from modules.scripts.data_preprocess import df_from_log
        df_raw = df_from_log(log_file=test_file, raw_data=True) # get only the raw data from the log file
        # display head of dataframe
        if debug_info:
                log("Raw data from log to determine timestamps:")
                print(df_raw.head())
        

        if not benign:
                # determine moments of detection which is an index of, only in the group of attack
                moments_of_detection = [k for k in range(len(true_anomalies)) if (true_anomalies[k]==1) & (attacked_signals[k]!=0)]

                if len(moments_of_detection)!=0:
                        start_of_attack = true_anomalies.index(1)

                        #print('Index of detection moment and start of actual attack')
                        t_start_of_attack = df_raw.index[start_of_attack]
                        t_start_of_detection = df_raw.index[moments_of_detection[0]]
                        time_to_detection =  t_start_of_detection-t_start_of_attack
                        if debug_info:
                                log('Time to detection: ' + str(time_to_detection))

                        anomalies = attacked_signals
                        anomalies.reverse()
                        index = anomalies.index(1)
                        last_detection = len(anomalies) - index - 1

                        '''if attacked_signals.index(1) != t_start_of_detection:
                                print(attacked_signals.index(1))
                                print(t_start_of_detection)'''

                        detected_at = t_start_of_detection

                else: 
                        time_to_detection = "not applicable"
                        last_detection = "not applicable"

                        log('Attack is undetected - ' + test_file[test_file.find('input/')+6:])


                
                first_flag = true_anomalies.index(1)
                
                reverse_anomalies = true_anomalies
                reverse_anomalies.reverse()
                index = reverse_anomalies.index(1)
                last_flag = len(reverse_anomalies) - index - 1

                # not attacked range
                first_range = range(first_flag)
                last_range = range(last_flag+evaluation_window_size, len(true_anomalies)) # +1 because range is exclusive at the end
                not_attacked_indexes = []
                not_attacked_indexes.extend(first_range)
                not_attacked_indexes.extend(last_range)
                # Extract elements from another_array using zero_indexes
                should_not_detect_indexes = [attacked_signals[index] for index in not_attacked_indexes]
                zeros = [true_anomalies[index] for index in not_attacked_indexes]

                

        else: #benign
                should_not_detect_indexes = np.maximum.reduce(binary_array)
                attacked_signals = np.maximum.reduce(binary_array)
        
        
        # Iterate through the binary array and count the 1s, false positive
        count_fp = 0
        for element in should_not_detect_indexes:
                if element == 1:
                        count_fp += 1
        # Calculate the false positive rate
        FPR = count_fp / len(should_not_detect_indexes)
        FPR_percentage = FPR * 100

        # -----------------------  Confusion matrix --------------------------
        from sklearn.metrics import confusion_matrix

        CM = confusion_matrix(true_anomalies, attacked_signals , labels=[0, 1])

        #log("Confusion matrix values")
        #print(CM)
        # shape and set of true anomalies and attacked signals 
        #log("Shape of true anomalies: " + str(len(true_anomalies)))
        #log("Shape of attacked signals: " + str(len(attacked_signals)))
        #log("Set of true anomalies: " + str(set(true_anomalies)))
        #log("Set of attacked signals: " + str(set(attacked_signals)))


        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]

        tn, fp, fn, tp = confusion_matrix(true_anomalies, attacked_signals, labels=[0, 1]).ravel()

        '''log("Confusion matrix values")
        log("TN, tn: " + str(TN) + ", " + str(tn))
        log("FN, fn: " + str(FN) + ", " + str(fn))
        log("TP, tp: " + str(TP) + ", " + str(tp))
        log("FP, fp: " + str(FP) + ", " + str(fp))'''



        
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import os
        from modules.scripts.config import work_dir
        import matplotlib.pyplot as plt

        if save:
                #model_dir = f"{work_dir}/evaluate/output/model_{model_config.get('model_id')}"
                output_path = f"{output_folder}/detections_on_{os.path.basename(test_file)[:-4]}"
                disp = ConfusionMatrixDisplay(confusion_matrix=CM)
                disp.plot()
                plt.savefig(f"{output_path}/confusion_matrix_attacked_group.png")
                plt.close()

        

        # -----------------------  Save Metrics --------------------------------------------------------
        if not benign:
                #log("DEBUG", str(set(true_anomalies) - set(attacked_signals)))
                i = attack_group
                metrics= {}
                metrics['accuracy'] = accuracy_score(true_anomalies, binary_array[i])
                metrics['precision'] = precision_score(true_anomalies, binary_array[i])
                metrics['recall'] = recall_score(true_anomalies, binary_array[i])
                metrics['f1_score'] = f1_score(true_anomalies, binary_array[i])
                metrics['first flag from log'] = first_flag
                metrics['time to detection'] = time_to_detection
                metrics['last detection'] = last_detection
                metrics['last flag from log'] = last_flag
                metrics['FPR'] = FPR
                metrics['FPR_percentage'] = FPR_percentage
                metrics['confusion matrix'] = CM.ravel().tolist()
                metrics['tpr'] = tp / (tp + fn)
                metrics['fnr'] = fn / (fn + tp)


                from IPython.display import display
                if debug_info:
                        display(metrics)

        # ------------------ metrics for other groups ----------------------

        
        # List of accuracy scores from multiple samples
        accuracy_scores = []
        if not benign:
                accuracy_scores.append(metrics['accuracy'])

        # sum of false positives from multiple samples
        fprs = []
        fprs.append(FPR) #fp, attacked part + 1 evaluation window not included
        tns = tn 
        fps = fp


        if benign:
                assert len(attacked_groups) == 0, "benign samples should not have attacked groups"
        
        # ignore these groups because they contain digital signals, detection process is set for analog signals
        ignored_groupes = [2,5,6,7,8] if len(binary_array) == 9 else [] #NOTE: [BASELINE] set back

        for group in range(len(binary_array)):

                # skip ignored groups
                if group in ignored_groupes or group in attacked_groups:
                        continue

                true_benign = np.zeros(len(binary_array[group]))
                #true_benign[0] = 1
                #binary_array[group][0]=1
                #binary_array[group][1]=0
                accuracy_scores.append(accuracy_score(true_benign, binary_array[group]))
                # no use to calculate precision on benign - precision_scores.append(precision_score(true_benign, binary_array[group]))
                # no use to calculate recall on benign - recall_scores.append(recall_score(true_benign, binary_array[group]))
                # no use to calculate f1 score on benign - f1_scores.append(f1_score(true_benign, binary_array[group]))
                tn, fp, fn, tp = confusion_matrix(true_benign, binary_array[group], labels=[0, 1]).ravel()
                #log("Confusion matrix values for benign traces: ", "tn: " + str(tn) + ", fp: " + str(fp) + ", fn: " + str(fn) + ", tp: " + str(tp))
                fpr = fp / (fp + tn)
                fprs.append(fpr)
                tns += tn
                fps += fp

        score_fpr = sum(fprs)/len(fprs)
        score_accuracy = sum(accuracy_scores)/len(accuracy_scores)
        score_tnr = tns / (tns + fps)

        # Calculate the mean accuracy
        mean_accuracy = score_accuracy
        mean_fpr_score = score_fpr
        mean_metrics = {}
        mean_metrics['mean accuracy'] = mean_accuracy
        mean_metrics['mean fpr'] = mean_fpr_score
        mean_metrics['mean tnr'] = score_tnr


        reported_metrics = {}
        reported_metrics['FPR'] = mean_metrics['mean fpr']
        if not benign:
                reported_metrics['attacked group confusion matrix'] = CM.ravel().tolist()
                reported_metrics['detected'] = 1 if  'not applicable' not in str(metrics['time to detection'])  else 0
                reported_metrics['time to detection'] = metrics['time to detection'] 
                #reported_metrics['roc_auc'] = [roc_auc]
                

        monitored_metrics = {}
        monitored_metrics['fpr'] =  mean_metrics['mean fpr'] #metrics['']


        if not benign:
                return metrics, reported_metrics, monitored_metrics, mean_metrics
        return {}, reported_metrics, monitored_metrics, mean_metrics

