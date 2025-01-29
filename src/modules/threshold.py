

# This script is for setting detection threshold for a new model


# Reading file paths from config file
from scripts.config import TrainedModels
from scripts.visualization import log
import tensorflow as tf
from tensorflow.keras.models import load_model
from tcn import TCN
from scripts.config import models_dir, input_data

import tensorflow_addons as tfa
from scripts.utils import get_name
import numpy as np
from scripts.evaluation import calculate_group_loss, calculate_summed_losses
import os
#pred, df_test,df_pred
from scripts.evaluation import multistep_predict
from glob import glob


# predict for benign file to analyse loss on normal behaviour
def predict_benign(model_id, benign_file):

    # Path to files and model
    tm = TrainedModels()
    # get config
    config = tm.get_config(model_id)

    # import stored model
    model_path = f"{models_dir}/model_{model_id}.h5"
    model = load_model(model_path, custom_objects={'WeightNormalization':tfa.layers.WeightNormalization, 'TCN': TCN})
    #model.summary()


    # get filenames for prediction file and loss file
    pred_filename = get_name(type_='predictions', model_id_=model_id, file_=benign_file, ext_='csv')
    loss_filename = get_name(type_='losses', model_id_=model_id, file_=benign_file, ext_='npy')

    # if there is no loss or pred file yet, make prediction for benign
    if not os.path.isfile(loss_filename) or not os.path.isfile(pred_filename):

        # make prediction
        _, df_benign, df_pred_benign = multistep_predict(model, config, benign_file, debug_info = False, plot = False)

        # calculate loss on benign prediction
        l_benign = calculate_group_loss(df_benign,df_pred_benign,config, plot = False)

        # store benign prediction in file
        df_pred_benign.to_csv(pred_filename, index=False)

        # store loss in file
        with open(loss_filename, 'wb') as f:
            np.save(f, l_benign)
        return l_benign
    
    # if we already did prediction on benign trace, load prediction and loss
    else:
        log("Found saved predictions and losses, reading from file.")
        #df_test, _ , _ = load_data(file_, config)
        #df_pred = pd.read_csv(pred_filename)
        losses = np.load(loss_filename)
        return losses

def calculate_threshold(model_id, benign_losses, window_size, sigma_rule=3):

    #first_thresholds = []
    #for loss in benign_losses:
    #    first_thresholds.append(np.mean(loss) + 3 * np.std(loss))

    # store first thresholds
    #training_config["thresholds"] = ['{:.8f}'.format(round(t,8)) for t in first_thresholds]
    #trainedModels.update(model_id, training_config)

    # sum above first threshold
    #areas_grouped = calculate_areas(benign_losses, model_id, window_size=window_size)
    areas_grouped = calculate_summed_losses(group_losses=benign_losses, window_size=window_size)
    

    
    return areas_grouped


def store_thresholds(model_id, thresholds, window_size, sigma_rule):

    # object that will store data to this model
    trainedModels = TrainedModels()

    # gather previously stored data for updating
    training_config = trainedModels.get_data(model_id)

    # add a new key-value pair with the thresholds as an array
    training_config["evaluation_window_size"] = window_size
    training_config["sigma_rule"] = sigma_rule
    training_config["area_thresholds"] = ['{:.8f}'.format(round(t,8)) for t in thresholds] # round thresholds a bit

    log("Calculated these thresholds", str(training_config["area_thresholds"]))

    # update stored data
    trainedModels.update(model_id, training_config)
        
        


def main(model_id, evaluation_window_size, sigma_rule=3):

    old_traces = input("Do you want to use old traces? (y/n): ")
    if old_traces == "y":
        benign_files = glob(f"{input_data}/old_data/benign_logs/files/trace_*.log")
        log(f"Using old benign data from directory {input_data}/old_data/benign_logs/files/")
    else:
        benign_files = glob(f"{input_data}/data/All_data/S*/*benign.log")
        log(f"Using new benign data from directory {input_data}/data/All_data/S*/")

                    

    areas_grouped_array = []

    for i,file in enumerate(benign_files):
        print("Analyzing benign file " + str(file))
        # make benign prediction with model
        benign_losses = predict_benign(model_id = model_id, benign_file = file)

        # calculate threholds using benign prediction
        areas_grouped_ = calculate_threshold(model_id=model_id, benign_losses=benign_losses, window_size=evaluation_window_size, sigma_rule=sigma_rule)
        if i == 0:
            areas_grouped_array = areas_grouped_
        else:
            # Append each sublist element-wise
            list1 = areas_grouped_array
            list2 = areas_grouped_
            result = [sublist1 + sublist2 for sublist1, sublist2 in zip(list1, list2)]
            areas_grouped_array = result

    
    assert len(areas_grouped_array) == len(benign_losses)

    # calculate threshold with 3-sigma rule
    thresholds = [] # this will store thresholds for each group
    for loss in areas_grouped_array:
        # use 4th standard deviation of the mean
        thresholds.append(np.mean(loss) + sigma_rule * np.std(loss))

    # store thresholds
    store_thresholds(model_id=model_id, thresholds=thresholds, window_size=evaluation_window_size, sigma_rule=sigma_rule)
    


if __name__ == "__main__":
    # read model id from input
    model_id = int(input("Enter model id number (no default): "))

    evaluation_window_size = int(input("Enter evaluation window size (no default): "))
    sigma_rule = int(input("Enter sigma rule (default 3): "))
    
    main(model_id=model_id, evaluation_window_size=evaluation_window_size, sigma_rule=sigma_rule)