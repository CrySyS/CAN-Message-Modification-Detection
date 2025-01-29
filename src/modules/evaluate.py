# Imports
# For model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tcn import TCN
import tensorflow_addons as tfa
# For data
import pandas as pd
import os
# For visualisation
from scripts.visualization import log, visualize_config
# Config params
from scripts.config import TrainedModels
from scripts.config import work_dir, models_dir, work_data, input_data, output_data
# For evaluation
from scripts.evaluation import multistep_predict, read_true_anomalies
from scripts.data_preprocess import load_data
from scripts.utils import get_name
# For detection
from scripts.evaluation import calculate_group_loss, area_detection, calculate_summed_losses
from scripts.utils import rolling_median
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# For metrics
from scripts.evaluation import calculate_metrics
import time
from matplotlib.lines import Line2D
from glob import glob
from tqdm import tqdm
import json
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scripts.utils import flatten
import matplotlib

debug_info = False

start = time.time()

folder_message = ""

# Test GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
log("Num GPUs:" + str(len(physical_devices)))

def main(model_id, detection_window, folder_message = "", d_info = False, save = True, skip_done=True):
    output_folder = f"{output_data}/evaluate_output/model_{model_id}{folder_message}"

    traces = input("Which traces do you want to evaluate on? (1: attacks (default) / 2: benign): ")
    if traces == "1" or traces == "":
        traces = "attacks"
    elif traces == "2":
        traces = "benign"
    else:
        raise Exception("Please enter the number of a valid trace type.")
    
    old_traces = input("Do you want to evaluate on OLD TRACES? (y / n (default)): ")
    old_traces = True if 'y' in old_traces else False

    

    old_model = input("Do you want to evaluate on OLD MODEL? (y / n (default)): ")
    old_model = True if 'y' in old_model else False

    # --------------------------- READ CONFIG AND FILES -----------------

    tm = TrainedModels()
    model_config = tm.get_config(model_id)
    #test_files = os.listdir(work_dir+'/evaluate/input/')
    #test_files = glob(f"{work_dir}/evaluate/new_figures_2/*.log")
    
    if traces == "attacks":
        if old_traces:
            test_files = glob(f"{input_data}/old_data/attacks/*/*.log")
            log(f"Evaluating on old attacled traces from directory: {input_data}/old_data/attacks/*/*.log")
        else:
            # ask for attack type
            attack_type = input("Which attack type do you want to evaluate on? ( 1: injection / 2: modification): ")
            if attack_type == "1":
                test_files = glob(f"{input_data}/evaluate_input/injection_attacks/*.log")
                log(f"Evaluating on new traces from directory: {input_data}/evaluate_input/injection_attacks/*.log")
            elif attack_type == "2":
                test_files = glob(f"{input_data}/evaluate_input/files_less/*.log")
                log(f"Evaluating on new traces from directory: {input_data}/evaluate_input/files_less/*.log")
            else:
                raise Exception("Please enter the number of a valid attack type.")
        
    elif traces == "benign":
        if old_traces:
            test_files = glob(f"{input_data}/old_data/benign_logs/files/*.log")
            log(f"Evaluating on old benign traces from directory: {input_data}/old_data/benign_logs/files/*.log")
        else:
            test_files = glob(f"{input_data}/evaluate_input/benign_files/*.log")
            log(f"Evaluating on new traces from directory: {input_data}/evaluate_input/benign_files/*.log")

    else:
        raise Exception("Can only evaluate on attacks or benign traces.")


    # dummy check for test file paths
    assert len(set([i[:10] for i in test_files])) == 1, "All test files must be from the same dataset. Please check the file names."

    if d_info:
        visualize_config(config=model_config)

    # --------------------------- LOAD MODEL ----------------------------


    model_path = f"{models_dir}/model_{model_id}.h5"
    model = load_model(model_path, custom_objects={'WeightNormalization':tfa.layers.WeightNormalization, 'TCN': TCN})

    if d_info:
        model.summary()

    # --------------------------- EVALUATE FILES ------------------------
    #metrics_for_files = {}

    fprs = []
    tprs = []
    tnrs = []
    fnrs = []
    accuracies = []
    times_to_detect = []
    detected = 0

    #test_files = ["S-3-3-malicious-REPLAY-msg-mod-0x410-0.4-0.6.log"]
    # Create a tqdm progress bar
    progress_bar = tqdm(enumerate(test_files), desc='Processing test files', total=len(test_files))
    for c, file_ in progress_bar:

        # skip already evaluated traces
        output_path = f"{output_folder}/detections_on_{os.path.basename(file_)[:-4]}"
        if skip_done and os.path.isdir(output_path):
            if d_info:
                log(f"file exits: {output_path}")
            log("Already evaluated, skipping trace " + file_)
            continue

        benign = True if 'benign' in file_ else False

        log(str(c)+". test file", file_)
        metrics, reported_metrics, monitored_metrics, mean_metrics = evaluate_file(file_ = file_, model= model, model_config=model_config, model_id = model_id, detection_window = detection_window, output_folder = output_folder, save = save, benign = benign, old_traces = old_traces, old_model=old_model)
        log(f"Monitored metrics \n{monitored_metrics}")

        fprs.append(mean_metrics['mean fpr'])
        tnrs.append(mean_metrics['mean tnr'])
        accuracies.append(mean_metrics['mean accuracy'])
        if not benign:
            tprs.append(metrics['tpr'])
            fnrs.append(metrics['fnr'])
            if  'not applicable' not in str(reported_metrics['time to detection']): 
                times_to_detect.append(reported_metrics['time to detection'])
            detected += reported_metrics['detected'] 

        #metrics_for_files[file_] = {'metrics':metrics, 'reported_metrics':reported_metrics, 'monitored_metrics':monitored_metrics, 'mean_metrics':mean_metrics}

    if not benign:
        # plot confusion matrix


        CM = np.array([[-1, -1], [-1, -1]])
        CM[0,0] = sum(tnrs)/len(tnrs)
        CM[0,1] = sum(fprs)/len(fprs)
        CM[1,0] = sum(fnrs)/len(fnrs)
        CM[1,1] = sum(tprs)/len(tprs)

        if save:
            disp = ConfusionMatrixDisplay(confusion_matrix=CM)
            disp.plot()
            plt.savefig(f"{output_folder}/confusion_matrix.png")
            plt.close()

            # plot distribution of times to detect and save to file
            for name, array in zip(['times_to_detect', 'accuracies'], [times_to_detect,  accuracies]):
                plt.hist(array)
                plt.xlabel(name.replace('_', ' ').capitalize())
                plt.ylabel("Frequency")
                plt.xticks(rotation=45)
                plt.savefig(f"{output_folder}/{name}.png")
                plt.close()

    
    # save all metrics
    report = {
        "files": test_files,
        "fpr": sum(fprs)/len(fprs),
        "accuracy": mean_metrics['mean accuracy']
    }
    if not benign:
        attack_metrics = {
            "tpr": sum(tprs)/len(tprs),
            "precision": metrics['precision'],
            "time to detect": sum(times_to_detect)/len(times_to_detect) if len(times_to_detect) > 0 else "not applicable",
            "detected": detected,
            "number of files": len(test_files),
            "detection rate": detected/len(test_files)
        }
        # concatenate two dictionaries
        report["attack_metrics"] = attack_metrics

    if save: 
        # convert report dictionary to string
        report_ = json.dumps(report, indent=4, sort_keys=True)
        # save repport to file
        with open(f"{output_folder}/report.json", 'w') as f:
            f.write(report_)

        
    return report

def evaluate_file(file_, model, model_config, model_id, output_folder, detection_window = 200, save = True, benign = False, old_traces = False, old_model = False):

    output_path = f"{output_folder}/detections_on_{os.path.basename(file_)[:-4]}"

    if benign:
        attack_name = "No attack"
    elif "malicious" in file_:
        attack_name = file_[file_.find('s-')+2:file_.find('-msg')].replace('_', '-')
    elif "trace" in file_:
        attack_name = file_.split('-')[2]
    else:
        log("Warning", "No attack name found in file name, using file name as attack name")
        attack_name = os.path.basename(file_)[:-4]
    

    # --------------------------- PREDICT ------------------------------
    test_file = file_ #f"{work_dir}/evaluate/input/{file_}"
    pred_filename = get_name(type_='predictions', model_id_=model_id, file_=os.path.basename(test_file), ext_='csv')
    #log(f"pred_filename {pred_filename}")
    if not os.path.isfile(pred_filename):
        log("No saved prediction, predicting now...")
        # Predict with model
        _, df_test,df_pred = multistep_predict(model, model_config, test_file, debug_info = d_info, plot = False)
        df_pred.to_csv(pred_filename, index=False)

    else:
        log("Found saved prediction, reading from file.")
        df_test, _ , _ = load_data(test_file, model_config)
        df_pred = pd.read_csv(pred_filename)




    signal_groups = model_config.signal_groups
    num_signals = sum([len(group) for group in signal_groups])
    num_groups = len(signal_groups)
    assert df_pred.shape[1] == num_signals

    # --------------------------- GROUP LOSSES ------------------------
    log("Calculating group losses...")
    # Calculate loss of prediction
    group_losses = calculate_group_loss(df_test, df_pred, model_config)

    # Rolling median on losses
    group_losses = rolling_median(group_losses, num_signals)

    # --------------------------- AREA LOSSES -------------------------
    log("Calculating evaluation window values for each timestamp ...")
    areas_grouped = calculate_summed_losses(group_losses = group_losses, window_size = detection_window)

    # --------------------------- DETECTION ----------------------------
    # detection_indexes_grouped is a list of lists, each list contains indexes of detections in a group (indexes where value of loss is higher than threshold)
    # all_array is a list of lists, each list contains either a value (loss/threshold percentage) of loss or 0 (if loss is lower than threshold)
    # binary_array is a list of lists, each list contains either a value (1) or 0 (if loss is lower than threshold)
    detection_indexes_grouped, all_array, binary_array = area_detection(areas_grouped, model_id, entire_array = True)

    # saving arrays for debug purposes
    if debug_info:
        for array, name in zip([detection_indexes_grouped,all_array, binary_array], ['detection_indexes_grouped','all_array', 'binary_array']):
            with open(f"{work_data}/temp/{name}.txt", 'w') as f:
                    # Write the 2D list to the file
                    for row in array:
                        # Join elements of each row with a delimiter (e.g., comma) and write to the file
                        f.write(','.join(map(str, row)) + '\n')


    assert len(detection_indexes_grouped) == num_groups

    # --------------------------- PLOT ANOMALIES ----------------------

    font = { 'size'   : 18}

    matplotlib.rc('font', **font)

    # colors for plotting
    original_color = 'teal'
    prediction_color = 'mediumblue'
    true_attack_color = 'gainsboro'
    detection_color = 'sandybrown'

    if save: 
        if not os.path.exists(output_path):
            # Create a new model directory because it does not exist
            log(f"Output directory does not exists, creating one: {output_path}")
            os.makedirs(output_path)

    thresholds = model_config.get("area_thresholds")

    detected_groups = []

    # -------- color map --------------------- 

    # NOTE: this next part should be used to get a color map from the min and max values of losses, now it is precalculated and hardcoded to be uniform in all plots
        # get all loss/threshold precentage values from all groups as a simple list
        # flatted_losses = flatten(all_array)
        # get max and min values from all occuring losses
        # max_percentage = max( flatted_losses )
        # min_percentage = min( flatted_losses)
    
    # get color map from yellow to red
    cmap = plt.get_cmap("YlOrRd") #ListedColormap(['springgreen','greenyellow','yellow','darkorange','orangered'])
    # normalizer between min and max
    norm = plt.Normalize(0, 85)

    # Determine which signal is attacks according to the test file name
    # NOTE: this has to be generalised to work with other datasets
    attack_in_signal = file_[file_.find('0x'):]
    attack_in_signal = attack_in_signal[:attack_in_signal.find('-')]
    attack_in_signal = attack_in_signal.replace('x', '') # attack_in_signal will be like '0410'

    if old_model:
        attacked_groups = [0]
        log("Old model, using attacked group [0]")
    elif '0110' in attack_in_signal:
        attacked_groups = [0,4] # for double attack
        log("Double attack, using attacked groups [0,4]")
    else:
        #raise Exception("Please set this according to which group contains the attack. (only needed for visualizatin, metrics are calculated independently)")
        attacked_groups = [4]
        log("Single attack, using attacked group [4]")

    if benign:
        attacked_groups = []
        log("Benign file, using no attacked group []")

    # ignore these groups because they contain digital signals, detection process is set for analog signals
    ignored_groupes = [2,5,6,7,8] if num_groups == 9 else [] #NOTE: [BASELINE] set back

    if save:
        for group_index, group in enumerate(detection_indexes_grouped):

            # skip ignored groups
            if group_index in ignored_groupes:
                continue

            if len(group) > 0 or group_index in attacked_groups:
                detected_groups.append(group_index)
    
                plt.figure(figsize=(12,6))
                #plt.title("Anomalies in group " + str(signal_groups[group_index]))
                plt.title(f"{attack_name}")

                

                max_value = max(flatten(df_test[signal_groups[group_index]].values))
                max_value = max_value*1.01
                min_value =  min(flatten(df_test[signal_groups[group_index]].values))
                min_value = min_value- max_value*0.11
                range_ = max_value - min_value
                # set y limit to max value
                plt.ylim(min_value, max_value)     

                y_min_percentage = 0.05

                # --------- TRUE ATTACK VERTICAL LINE
                if group_index in attacked_groups: #NOTE: this has to be generalised to work with other attacks
                    
                                  
                    # normalize max_value between 0 and 1 

                    true_anomalies = read_true_anomalies(test_file)
                    anomaly_indexes = [ind for ind, x in enumerate(true_anomalies) if  x == 1]
                    for anomaly_index in anomaly_indexes:
                        plt.axvline(anomaly_index, ymin = y_min_percentage, ymax=1, color=true_attack_color, zorder=0)
                

                # --------- DETECTION VERTICAL LINES
                # plot vertical lines for each detection index
                for index in group:
                    # normalize current detection value and get a color from colormap
                    color = cmap(norm(all_array[group_index][index]))
                    plt.axvline(index, ymin=y_min_percentage, ymax=1, alpha=1, color=color)
                
                
                # -------- TRUE ATTACK HORIZONTAL RECTANGLE
                if group_index in attacked_groups: #NOTE: this has to be generalised to work with other attacks
                    # plot another indicator line for true anomalies on top of the plot, a thin horizontal green line at the bottom
                    # get the first and the last index of 1s in the true anomalies array
                    first_anomaly = np.argmax(true_anomalies)
                    last_anomaly = len(true_anomalies) - np.argmax(np.flip(true_anomalies))
                    #plt.hlines(y=0, xmin=first_anomaly, xmax=last_anomaly, color=true_attack_color)
                    
                    rec_y_min = min_value
                    rec_y_max = y_min_percentage
                    # plot white rectangle from x = 0 to x = first_anomaly, and y = 0 to y = 0.1
                    #plt.axvspan(xmin=0, xmax=first_anomaly, ymin=rec_y_min, ymax=rec_y_max, color='white')
                    # plot true_attack_color rectangle from x = first_anomaly to x = last_anomaly, and y = 0 to y = 0.1
                    plt.axvspan(xmin=first_anomaly, xmax=last_anomaly, ymin=rec_y_min, ymax=rec_y_max, color=true_attack_color)
                    # plot white rectangle from x = last_anomaly to x = len(true_anomalies), and y = 0 to y = 0.1
                    #plt.axvspan(xmin=last_anomaly, xmax=len(true_anomalies), ymin=rec_y_min, ymax=rec_y_max, color=true_attack_color)

                    #rectangle = patches.Rectangle((first_anomaly, rec_y_min), last_anomaly-first_anomaly,rec_y_max, facecolor='blue', zorder=1)
                    #plt.gca().add_patch(rectangle)
                

                # -------- ORIGINALS AND PREDICTIONS
                # plot first signal separately to avoid multiple labels 
                plt.plot(df_test[signal_groups[group_index][0]], color=original_color, label="Original")
                plt.plot(df_pred[signal_groups[group_index][0]], color=prediction_color, label='Prediction')
                if len(signal_groups[group_index]) > 1:
                    plt.plot(df_test[signal_groups[group_index][1:]], color=original_color)
                    plt.plot(df_pred[signal_groups[group_index][1:]], color=prediction_color)

                # add axes labels
                plt.xlabel("Timestamps")
                plt.ylabel("Normalized signal value")
                            

                img_name = 'group'+str(group_index)+'__'+str(signal_groups[group_index]).replace('[', '').replace(' ', '').replace(']','').replace("'", '').replace(',', '__')
                img_filename = f"{output_path}/{img_name}"
                
                # set labels for the plot
                custom_lines = [Line2D([0], [0], color=original_color, label = 'Signals', lw=2),
                                Line2D([0], [0], color=prediction_color, label = 'Predictions', lw=2),
                                #Line2D([0], [0], color='tomato', label="threshold", linestyle='--', lw=2),
                                Line2D([0], [0], color=true_attack_color, label = 'Attack', lw=2, marker='|', linestyle='None', markersize=10, markeredgewidth=2),
                                Line2D([0], [0], color=detection_color, label = 'Detection', lw=2, marker='|', linestyle='None', markersize=10, markeredgewidth=2)]
                plt.legend(handles=custom_lines, loc='upper left')


                plt.savefig(f"{img_filename}.png")
                plt.savefig(f"{img_filename}.svg")

                matplotlib.pyplot.close()

            
            
    # --------------------------- METRICS ------------------------------
    # anomaly indexes, first recognition, and metrics
    metrics, reported_metrics, monitored_metrics, mean_metrics = calculate_metrics(test_file=test_file, model_config=model_config, binary_array=binary_array, evaluation_window_size= detection_window, output_folder = output_folder, save=save, benign = benign, debug_info = False)
    log("Metrics calculated")
    log(f"detected in groups: {detected_groups}")
    
    if save:
        # save all metrics
        for name, dict in zip(['metrics', 'reported_metrics', 'monitored_metrics', 'mean_metrics'], [metrics, reported_metrics, monitored_metrics, mean_metrics]):

            with open(f"{output_path}/{name}.json", 'w') as f:
                # write json to file with indent and formatted
                # debug info for each metric
                #for key, value in dict.items():
                    #print(f"Key: {key}, Type: {type(value)}")
                json.dump(dict, f, indent=4, sort_keys=True)

    return metrics, reported_metrics, monitored_metrics, mean_metrics






if __name__ == "__main__":

    # --------------------------- CHOOSE MODEL --------------------------
    log("Evaluate a trained model", "Read input test files from work_dir/evaluate/input/\nWrites images to work_dir/evaluate/output/")
    model_id = int(input("Enter model id number: "))
    folder_message = input("Do you want to add message to the output folder's name?: " or "")
    #d_info = input("Evaluate with debug info? (y / default: n): ")
    #d_info = True if 'y' in d_info else False
    d_info = False
    #skip_done = input("Do you want to skip already evaluated traces? (default: y / n): ")
    #skip_done = False if 'n' in skip_done else True
    skip_done = True
    
    detection_window = int(input("Enter detection window size (default 200): ") or 200)

    

    # --------------------------- EVALUATE FILES ------------------------
    main(model_id=model_id, detection_window=detection_window, d_info = d_info, folder_message = folder_message,  skip_done=skip_done)


    
    end = time.time()
    elapsed = end - start
    log("time elapsed in executing evaluation", time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15], time.gmtime(elapsed)))