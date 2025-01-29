
from modules.scripts.config import Config, work_dir
import tensorflow as tf
from modules.scripts.visualization import log
import numpy as np
from modules.scripts.data_preprocess import load_data_part, load_data
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd

# ---------------- Datagenerator for multiple input files --------------
# This generator is used to train on multiple traces, without overlapping between them
# not implemented functions:
#       - stride, because we did not use it
#       - shuffle (shuffle can be done as far as the file reading is concerned, but files are handled in a sequential way, it should be updated (choose the correct file for the index if index can be shuffled))


# only stores one file at a time in the memory
class MultipleFileGenerator(tf.keras.utils.Sequence):

    # initialise everything, and first generator
    def __init__(self, file_paths, window_size, batch_size, config, part: Tuple[float, float], debug_info = False):

        # set which part of file to use (for example 0-80% for training)
        self.part_from, self.part_to = part
        # ----------- other info -----------------------------
        # store whether we need to pring debug info later
        self.debug_info = debug_info
        #log(f"Setting up datagenerator {for_what}",f"Files: {[str(f) for f in file_paths]}", emphasize=True)
        if debug_info:
            log(f"Setting up datagenerator from {self.part_from} to {self.part_to} part of data", emphasize=True)

        # ------------ params for generating data -------------

        # batch size and window size for each file
        self.batch_size = batch_size
        self.window_size = window_size
        # paths of each file
        self.file_paths = file_paths
        # lengths of each file will be stored in this array
        self.data_lengths =[]
        self.file_lengths =[] # only for checking if not reading too much
        # training config for loading each file
        self.config = config

        self.loaded_data = []
        
        # ------------- calculate lengths of given files ------
        # open all files and determine the lengths of each, without loading any relevant data
        for file in tqdm(file_paths):
            

            # storing lengths of each file and iterable lengths of each file
            with open(file, 'r') as fp:
                lines = len(fp.readlines())

                length_of_file = lines
                
                used_percentage = self.part_to - self.part_from # training: (0, 0.8) -> 0.8, validation (0.8,1) -> 0.2
                length_of_file = int(lines*used_percentage)

                iterable_size = (length_of_file-self.window_size)//self.batch_size + 1 # +1 is for the last batch, which won't be full
                

                self.data_lengths.append(iterable_size)
                self.file_lengths.append(lines)
            
            # check if file is probably not too large to load to memory
            if length_of_file < 10**5:
                # load data to memory
                data, _, _ = load_data(filename=file, config=self.config, debug_info=self.debug_info)
                self.loaded_data.append(data.values)

                if self.debug_info:
                    # log debug info about size of data
                    log(f"Loaded data, data shape is {data.shape}")
                    # size of data in MB is, formatted to 2 decimal places
                    log(f"Size of data is {data.memory_usage(deep=True).sum()/1024**2:.2f} MB")
                
            else:

                log("File is probably too large to read, will load only one batch at a time")
                self.loaded_data = None

            

    # return a batch of input data from the preprocessed file        
    def get_batch_from_preprocessed_file(self, index, file_index) -> List[List[float]]:

        filename = self.file_paths[file_index]

        # offset for generating validation data, if for_train is set to True, than this will be 80% of the current data length
        offset = int(self.file_lengths[file_index]*self.part_from)

        target_size = 1 # only setting this to allow multiple target sizes in the future
        # determine where to start reading in the file
        read_from = int(offset + index*self.batch_size)
        remaining_length = self.file_lengths[file_index] - read_from 
        read_to = min(self.batch_size+self.window_size+target_size-1, remaining_length) # in the last batch, there can be less than a full batch of data
    

        if self.debug_info:
            log(f"""\n\tReading from {self.part_from} to {self.part_to} (offset is {offset}),
                    \t\tfrom {read_from} to plus {read_to}
                    \t\tcurrent file total length is {self.file_lengths[file_index]} (file index is {file_index})
                    \t\tcurrent file iterable length is {self.data_lengths[file_index]} (iterator is {index})
                    \t\tall file lengths {self.file_lengths}
                    \t\tall iterable lenghts {self.data_lengths}
                    \t\tbatch is {self.batch_size}\n""")
        
        # this shouldn't be more, only testing if it works correctly
        lower_than_file_size = read_from+read_to < self.file_lengths[file_index] + 1

        if lower_than_file_size: 
                data:List[List[float]] = load_data_part(filename=filename, read_from=read_from, read_to=read_to, config=self.config, debug_info=False)#self.debug_info)
                
                if self.debug_info:
                    # log debug info about size of data
                    debug = f"\nLoaded batch of data, data shape is {data.shape}"
                    # size of data in MB is, formatted to 2 decimal places
                    debug += f" Size of data is {pd.DataFrame(data).memory_usage(deep=True).sum()/1024**2:.2f} MB"
                    # debug to file
                    with open(f"{work_dir}/debug.txt", "a") as f:
                        f.write(debug)
                
        else:
            log("Error loading file part", f"\n\tReading from {self.part_from} to {self.part_to} (offset is {offset}),\n\t\tfrom {read_from} to {read_to} (iterator is {index})\n\t\tcurrent file total length is {self.file_lengths[file_index]}\n", emphasize=True)
            raise Exception("Trying to read more than total file, ")
        
        #... further preprocess if needed (raw logs are already preprocessed when calling functions from data_preprocess.py)
        
        return data
    
    def get_batch_from_loaded_file(self, index, file_index) -> List[List[float]]:

        # offset for generating validation data, if for_train is set to True, than this will be 80% of the current data length
        offset = int(self.file_lengths[file_index]*self.part_from)

        target_size = 1 # only setting this to allow multiple target sizes in the future
        # determine where to start reading in the file
        read_from = int(offset + index*self.batch_size)
        remaining_length = self.file_lengths[file_index] - read_from 
        read_to = min(self.batch_size+self.window_size+target_size-1, remaining_length) # in the last batch, there can be less than a full batch of data
        
        return self.loaded_data[file_index][read_from:read_from+read_to]

    # this function will initialise the generator for the new epoch, starting data generation from the beginning
    def on_epoch_end(self):
        #log("Calling on epoch end, currently not doing anything")
        pass
        
        
    # return the length of all files together, this will determine the range in which the generator is generating (calling getitem)
    # so if this return 8, the generator can be called 8 times, so we need all rows counted
    def __len__(self):
        length = sum(self.data_lengths) if "SynCAN" not in self.file_paths[0] else min(10**3, sum(self.data_lengths)) # we do not want to train on all SynCAN data
        if self.debug_info:
            log(f"Length of datagenerator is {length}")
        return length
    
    # determine which file's data to use for this index
    def get_file_index(self, idx):
        if self.debug_info:
            log(f"Getting file index for iterator {idx}", f"iterables lenghts are {self.data_lengths}")
        
        passed_lenghts = 0
        
        # enumerate stored iterable lengths of files
        for file_index, iterable_length in enumerate(self.data_lengths):
            
            # if given index is less than the length, we are indexing that file
            if idx < iterable_length + passed_lenghts:
                if self.debug_info:
                    log(f"returning file index {file_index}")
                    log(f"idx is {idx} < (iterable length {iterable_length} + passed lenghts {passed_lenghts})")
                return file_index
            
            passed_lenghts += iterable_length

    # return current batch from current generator (generator is already stored with all batches, get specific (idx) element)
    def __getitem__(self, idx):
        if self.debug_info:
            log(f"get item for index {idx}")
            # log current memory usage
            import psutil
            log(f"Current memory usage is {psutil.virtual_memory().percent}%")

            import nvidia_smi

            nvidia_smi.nvmlInit()

            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            print("Total memory:", info.total/1024**2, ' MB')
            print("Free memory:", info.free/1024**2, ' MB')
            print("Used memory:", info.used/1024**2, ' MB')

            nvidia_smi.nvmlShutdown()

            # with tf.config.experimental.get_memory_info('GPU:0') in MB print total, free and used
            log(f"Current memory usage on GPU 0 is {tf.config.experimental.get_memory_info('GPU:0')['current']/1024**2} MB")
            log(f"Current memory usage on GPU 1 is {tf.config.experimental.get_memory_info('GPU:1')['current']/1024**2} MB")


 
        # determine which file to use for this index
        file_index = self.get_file_index(idx)

        # calculate the offset of the idx, so if we processed 2 files, with the length of 100 each, we have an idx 200+, the remaining part is relevant in current file
        processed_file_length = sum(self.data_lengths[ : file_index])

        # check whether data was loaded to memory during inicialisation
        if self.loaded_data is not None:
            # could load data to memory, using that instead of reading from file
            X = self.get_batch_from_loaded_file(index=idx-processed_file_length, file_index=file_index)
            if self.debug_info:
                log(f"Loaded batch from memory, data shape is {X.shape}")
        else:
            # data is not loaded to memory, reading from file
            X = self.get_batch_from_preprocessed_file(index=idx-processed_file_length, file_index=file_index)
            if self.debug_info:
                log(f"Loaded batch from file, data shape is {X.shape}")
        
        #X = X['data'] # only for tfrecord structure

        signal_groups = self.config.get("signal_groups")
        num_signals = sum([len(signal_groups[group_number]["signals"]) for group_number in signal_groups])

        # Prepare input sequences (train) and target sequences (target)
        train_sequences = np.reshape([X[i:i+self.window_size] for i in range(len(X)-self.window_size)], (-1,self.window_size,num_signals))
        target_sequences = np.reshape([X[i+self.window_size] for i in range(len(X)-self.window_size)], (-1,num_signals))

        if len(train_sequences) != len(target_sequences):
            log("Train and target length don't match", 
                f"""Train sequence length is {len(train_sequences)}
                    \nTarget sequence legth is {len(target_sequences)}""", emphasize=True)
            raise Exception("Can't yield enough train or target data.")

        # return the right element from current dataset
        return train_sequences, target_sequences