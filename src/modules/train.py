import tensorflow as tf
from scripts.config import Config
from scripts.visualization import visualize_config, log
import glob
# generating data
from scripts.datagenerator import MultipleFileGenerator
# TCN model
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from tcn import TCN
# training
from keras.callbacks import EarlyStopping
# save data about model
from scripts.config import TrainedModels
import datetime
# save model and train history
from scripts.config import input_data, output_data
import json
import time

# to measure elapsed time during training
start = time.time()

# Test GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
log("Num GPUs:" + str(len(physical_devices)))

# --------------------------- CHOOSE CONFIG FOR TRAINING --------------------------
conf_number = input("Enter config number: ")
additional_note = "" + input("Give additional note to this training: ")
debug_info = input("Train with debug info (y/n): ").lower().strip() == 'y'

assert int(conf_number) > 0

# Reading file paths from config file
config = Config(conf_id = f"conf{conf_number}")
if debug_info: visualize_config(config=config.get_config())

# --------------------------- FILE PATHS AND PARAMS FROM CONFIG ------------------------------------------


config.get_config()
signal_mask = None

train_folder = config.get("train_folder")
file_paths = glob.glob(f"{input_data}/{train_folder}/*")# /preprocessed_data/S-*.txt"):

trained_on = f"80% train, 20% validation: {file_paths}"
# --------------------------- Create data generator for sliding window ------------

log("Training on multiple files, generating with new generator", emphasize=True)

#stride = config.get("stride")
window_size = config.window_size
batch_size = config.batch_size
#target_values = config.get("multistep")

X_train_window = MultipleFileGenerator(file_paths=file_paths, window_size=window_size, batch_size=batch_size, config=config, part=(0,0.8), debug_info=debug_info)
X_validation_window = MultipleFileGenerator(file_paths=file_paths, window_size=window_size, batch_size=batch_size*2, config=config, part=(0.8,1), debug_info=debug_info)
#X_train_window = SingleFileGenerator(file_path=file_paths[0], confnumber=conf_number, debug_info=debug_info)
#X_validation_window = SingleFileGenerator(file_path=file_paths[1], confnumber=conf_number, debug_info=debug_info)


# ------------------------------ MODEL ------------------------------------
# set tensorflow to use all available GPU devices
strategy = tf.distribute.MirroredStrategy(devices=None)
# build model in this scope, so that it will be replecated and syncronized across all devices
with strategy.scope():
        log("Building model", emphasize=True)

        # Required params from config
        signal_groups = config.signal_groups
        num_signals :int = sum([len(group) for group in signal_groups])

        # Setting batch size, timesteps and input dimension
        batch_size, timesteps, input_dim = None, window_size, num_signals

        # Input layer
        i = Input(shape=(timesteps, input_dim), batch_size=batch_size)

        TCN_layers = []

        # create a TCN layer for each signal groups
        group_start = 0
        for group_index, group in enumerate(signal_groups):
                units = config.unit_groups[group_index]

                assert isinstance(units, int)
                assert isinstance(group, list)

                # choose the appropriate part of the input layer
                x = i[ :, : , group_start: group_start+len(group)]
                # create TCN layer on top of it
                x = TCN(nb_filters = units, kernel_size=16, nb_stacks = 1, dilations=[1,2,4], use_skip_connections=True, return_sequences=False)(x)
                        
                x = Dense(len(group))(x)

                TCN_layers.append(x)

                group_start += len(group)
        
        # Combine all TCN layers for an output layer
        combinedInput = tf.keras.layers.Concatenate(axis=-1)([x for x in TCN_layers])
        x = combinedInput


        # Create and compile model
        model = Model(inputs=i, outputs=x)
        model.compile(optimizer='adam', loss='mse',metrics = ['mae', 'mse', 'accuracy'])
        model.summary()


# Earlystopping callback for training
earlystop_callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.00001, mode='min', patience = 5, verbose = 1, restore_best_weights = True)

#  ------------------------------ TRAIN THE MODEL ------------------------------

log("Start training...", emphasize=True)

# Params from config
epochs = config.get("epochs")

# train model in this scope, so that gradients will be syncronized across all devices
with strategy.scope():
        # Train
        history = model.fit(x = X_train_window, validation_data=X_validation_window, epochs=epochs, verbose=1, callbacks=[earlystop_callback], shuffle=True)

# list all data in history
print(history.history.keys())
log(f"Training restored {earlystop_callback.best_epoch}. epoch", emphasize=True)
log(f"Val loss of best epoch was: {history.history['val_loss'][earlystop_callback.best_epoch]}")
log(f"Val MSE of best epoch was: {history.history['val_mse'][earlystop_callback.best_epoch]}")
log(f"Val accuracy of best epoch was: {history.history['val_accuracy'][earlystop_callback.best_epoch]}")

end = time.time()
elapsed = end - start
log("Time elapsed in executing training", time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15], time.gmtime(elapsed)))

# ----------- Save data about training in trained models yaml ------------------------------

time = datetime.datetime.now()
trainedModels = TrainedModels()

# gather data for storing
training_config = config.get_config().copy()
training_config["trained_on"] = trained_on
training_config["best_epoch"] = earlystop_callback.best_epoch
#training_config["val_mse_of_best_epoch"] = history.history['val_mse'][earlystop_callback.best_epoch]
training_config["time_of_training"] = time.strftime("%Y.%m.%d_%H:%M_")
training_config["training_time"] = time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15], time.gmtime(elapsed))
training_config["type"] = "One signal TCN" if num_signals == 1 else "Grouped multi-signal TCN"
training_config["additional_note"] = additional_note

# store data
last_id = trainedModels.get_last()
model_id = last_id + 1
trainedModels.store(model_id, training_config)

# ------------ Save model and history in files ----------------------------------------------

output_folder = f"{output_data}/models"
model_output_path = f"{output_folder}/model_{model_id}.h5"
history_output_path = f"{output_folder}/model_{model_id}_history.json"

model.save(model_output_path)
json.dump(history.history, open(history_output_path, 'w'))

log(f"Trained model {model_id}", emphasize=True)

