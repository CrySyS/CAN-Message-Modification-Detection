# CAN-Message-Modification-Detection
CAN anomaly detection with correlation-based signal clustering.


Read the full paper here: https://static.crysys.hu/v1/publications/files/KoltaiGA2024ICISSP

If you plan to use this work for your own research, please cite:
```
@inproceedings {
    author = {B. Koltai and A. Gazdag and G. Ács},
    title = {Supporting CAN Bus Anomaly Detection With Correlation Data},
    booktitle = {Proceedings of the 10th International Conference on Information Systems Security and Privacy - ICISSP},
    year = {2024}
}
```
## Used Dataset

The dataset used in this work could be found here: https://springernature.figshare.com/collections/CrySyS_dataset_of_CAN_traffic_logs_containing_fabrication_and_masquerade_attacks/6726165/1

You can read more in the topic here: https://www.crysys.hu/research/vehicle-security

## 1. **Introduction**
  > This project addresses the need for security in vehicular communication networks by developing a method for an Intrusion Detection System (IDS), which would detect anomalies in the CAN bus. Using a combination of time-series forecasting and signal correlation analysis, the system identifies potential attacks on Electronic Control Units (ECUs) that could compromise vehicle safety. The solution is implemented in Python using Keras, and it achieves significant improvements over existing methods in terms of detection accuracy: we achieve a detection rate of 95% (compared to 68%) with a precision of 80% (versus 30%). Additionally, our method exhibits a minimal average detection delay of just 0.38 seconds

## 2. **Features**
Key features of the system:
- Time-series forecasting and correlation analysis for anomaly detection
- Detects both message modification and injection attacks
- Operates on unlabeled CAN bus data for unsupervised learning
- Efficient preprocessing and clustering of CAN signals

## 3. **Technologies Used**
Important libraries and frameworks used in the project:
  - **Bitstring**: For encoding/decoding binary representations of CAN data.
  - **Pandas**: Data manipulation and handling CAN signal data.
  - **NumPy**: Mathematical operations and transformations on training data.
  - **Scikit-Learn**: Correlation analysis, clustering, and evaluation metrics.
  - **Keras (with keras-tcn)**: Building and training Temporal Convolutional Network (TCN) models.
  - **Matplotlib**: Visualizing training and evaluation results.

## 4. **How It Works**
The model preprocesses CAN logs by extracting and normalizing signals and applying correlation analysis to group similar signals. It then trains TCN models on these groups to detect abnormal behavior based on predefined thresholds. The model's performance is evaluated on benign and attack datasets, achieving a 95% detection rate, with a precision of 80%.


## 5. **Usage**

Install requirements from `requirements.txt`.

After downloading the dataset (mentioned earlier), and setting some path parameters (mostly in `src/modules/scripts/config.py`) you can run the individual modules for the whole process.

0) Use the `Preprocess - Clustering signals.ipynb` to analyse the data and set the clusterings of signals. We already calculated this for the dataset, so it is set in the configs.
1) Use `python train.py` to train a model.
2) Use `python threshold.py` to calculate and set a detection threshold for the newly trained model.
3) Use `python evaluate.py` to evaluate the working detection model on a test data set.

There are some additional modules, mostly for supplementray analyis.
- Run the `evaluation_window_analysis.py` to generate an animated gif which visualizes different window sizes, could be useful for setting this parameter.
- Run the `hyperparameter_tuning.py` if you want to calculate the best window size and sigma rule for the model from a given range.
- `SynCAN.ipynb` demonstrates the correlation analysis on another dataset, the SynCAN dataset.

### 5.1 Training a model

The file `train.py` is responsible for training a model from the dataset as described in the paper.

Configuration for training is stored in `train_config.yaml` file. You can add the followings:
- train_folder: folder with the input data for training
- description: some human readable summery of the specific training config
- window_size: parameter of the model, as described in the paper
- multistep: parameter of the input data processing, we recommend to use no multistep, so set it to 1
- stride: parameter for the input data processing, after experiments, we found that using stride did not improve the results, only helped with efficiency, so we set it to 1 (no stride)
- batch_size: batch size of the input
- epochs: maximum number of epochs, used with early stopping
- signal_groups: a dictionary, containing the groupes of the signlas, which are precalculated with the correlation module (see train_config.yaml file for example)

There is a `Config` class in the `scripts/config.py`, which handles these parameters for the training and other modules as well.

After that, a `MultipleFileGenerator` class is used from `scripts/datagenerator` which can yield one batch at a time from the files. This is useful if you have limited memory.

After that, a complex model structure is trained, which uses one TCN model for each clusters (signal groups).

The `trained_models.yaml` file will store information about each trained model, containing the configs as well. Later steps will write additional info here.
The `TrainedModels` class from `config.py` is responsibly for handling this file.

### 5.2 Threshold

A newly trained model needs a detection threshold to be set, which can be done with running the `threshold.py` script. It calculates the noise on the benign data, then calculates threshold for each signal group, and stores these values in the `trained_models.yaml` file.

### 5.3 Evaluation

The script `evaluation.py` could be used for testing the models with a given threshold set. It will generate a report with the results in the given output folder, as a json file.
Results for previous runs is stored in the `results.txt` and `results_injection.txt` files (separate tests for modification attacks and injection attacks, as mentioned in the paper).


### Signal extraction

To use the signals from the original CAN data, a signal mask is used from the `signal_extraction` folder. For ease of use, and understanding, the signal mask is exported in csv, which shows a line for each signal, and the start and end in the message. This start and end was reduced for some efficiency, and to not train on stand-by messages. This is why there is a second signal mask `src/signal_extraction/signal_mask_reduced_modified_for_0110_3.h5` which is used in the code.