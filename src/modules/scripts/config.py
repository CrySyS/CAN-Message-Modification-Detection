import yaml
from typing import Dict, List, Union, Optional

# Configurations for all files

# These folders will be used
git_basefolder = "..." # basefolder
work_dir = "_work_dir" # data basefolder (logs, models, produced data, etc.)
models_dir = f"{work_dir}/3_output_data/models" # models folder
work_data = f"{work_dir}/2_work_data" # predictions folder
input_data = f"{work_dir}/1_input_data" # input data folder
output_data = f"{work_dir}/3_output_data" # output data folder

# Path to signal mask
# NOTE: change this, if there's another new signal mask
signal_mask_path_old_traces = f"{git_basefolder}/src/signal_extraction/signal_mask_reduced_modified_for_0110_3.h5" # modified new signal mask for 0110_3, but positions was changed to 0110_1 bc old traces sometimes were constant 0 there
signal_mask_path= f"{git_basefolder}/src/signal_extraction/signal_mask_reduced.h5"
interesting_columns = set(['0110_0', '0110_1', '0110_3', '0120_0', '0120_1', '0120_2', '0120_3', '0280_0', '0280_1', '0280_2', '0280_3', '0290_0', '0290_4', '0290_1', '0290_2', '0300_4', '0381_6', '0381_4', '0410_0', '0410_4'])  


class Config:

    def __init__(self, conf_id: Optional[str]=None, conf_data = None ) -> None:

        if conf_id is not None and conf_data is not None:
            raise BaseException("Please use Config class with either a configuration id or the configuration data itself (dont set both conf_id and conf_data)")
        
        if conf_id is not None:
            self.path = git_basefolder + '/train_config.yaml'

            # Reading from config file
            with open(self.path, "r") as f:
                conf_file = yaml.load(f, Loader=yaml.FullLoader)
                self.config = conf_file[conf_id]
        
        if conf_data is not None:
            self.config = conf_data

        self.train_folder: str = self.config["train_folder"]
        self.description: str = self.config["description"]
        self.window_size: int = self.config["window_size"]
        self.multistep: int = self.config["multistep"]
        self.stride: int = self.config["stride"]
        self.batch_size: int = self.config["batch_size"]
        self.epochs: int = self.config["epochs"]

        signals_dict = self.config["signal_groups"]
        self.signal_groups: List[List[str]] = [signals_dict[key]["signals"] for key in signals_dict]
        self.unit_groups: List[List[int]] = [signals_dict[key]["units"] for key in signals_dict]
    
    # Return a configuration data (like stride, epochs, etc.)
    def get(self, key: str) -> Union[int, str, Dict[int, Dict[str, Union[List[str], int]]]]:
        return self.config[key]

    # Return used configuration
    def get_config(self) -> Dict[str, Union[str, int, Dict[int, Dict[str, Union[List[str], int]]]]]:
        return self.config

class TrainedModels:
    def __init__(self) -> None:

        # Storing path for updating
        self.path = git_basefolder + '/trained_models.yaml'

        # Reading saved training info
        with open(self.path, "r") as f:
            self.trained_models = yaml.load(f, Loader=yaml.FullLoader)
        
        # Storing last trained model id
        self.last_trained = self.trained_models["last_id"]
    
    # Return last trained model id
    def get_last(self):
        return self.last_trained
    
    # Return data about a model (id)
    def get(self, id, key):
        return self.trained_models['models'][id][key]

    # Return all data about a model (id)
    def get_data(self, id):
        return self.trained_models['models'][id]

    # Return config of a model
    def get_config(self, id):
        return Config(conf_data=self.trained_models['models'][id])

    # Add newly trained model info, save file
    def store(self, new_id, dictionary):
        self.trained_models["models"][new_id] = dictionary

        self.trained_models["last_id"] = new_id

        # Updating file
        # deepcode ignore BinaryWrite: <please specify a reason of ignoring this>
        with open(self.path, 'w') as file:
            yaml.dump(self.trained_models, file)

    # Update data
    def update(self, id, dictionary):
        self.trained_models["models"][id] = dictionary

        # Updating file
        with open(self.path, 'w') as file:
            yaml.dump(self.trained_models, file)
