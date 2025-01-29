# Imports
# For model
import tensorflow as tf
import seaborn as sns
import json
# For visualisation
from scripts.visualization import log
# Config params
from scripts.config import work_dir

# For detection
import matplotlib.pyplot as plt
# For metrics
import time
from evaluate import main as evaluate_main
from threshold import main as threshold_main
start = time.time()
from tqdm import tqdm

# Test GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
log("Num GPUs:" + str(len(physical_devices)))



if __name__ == "__main__":


    

    # --------------------------- CHOOSE MODEL --------------------------
    log("Evaluate a trained model", "Read input test files from work_dir/evaluate/input/\nWrites images to work_dir/evaluate/output/")
    model_id = int(input("Enter model id number: "))
    folder_message = input("Do you want to add message to the output folder's name?: " or "")
    
    d_info = input("Evaluate with debug info? (y/n): ")

    d_info = True if 'y' in d_info else False

    evaluation_window_size_range = range(1,501,50)
    sigma_rule = [4]

    log("Hyperparameter tuning", "Evaluating model with different evaluation window sizes and sigma rules")
    log(f"Evaluation window size range {evaluation_window_size_range}")
    log(f"Sigma rule range {sigma_rule}")


    times_to_detect=[]

    # Define a simple classes to represent the metric and hyperparameters
    class Monitored_metric:
        def __init__(self, name, is_max):
            self.name = name
            self.is_max = is_max

    class Hyperparameters:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    # ------------------------------------------------------- #
    # Set this to the metric you want to monitor and
    # whether you want to maximize or minimize it
    monitored_metric = Monitored_metric(name='detected', is_max=True) # name should be same as in report from evaluate.py
    # ------------------------------------------------------- #

    # define named tuples to store the best hyperparameters
    best_evaluation_window_size = Hyperparameters(name='evaluation window size', value=-1)
    best_sigma_rule = Hyperparameters(name='sigma rule', value=-1)
    hyperparams = [best_evaluation_window_size, best_sigma_rule]
    current_best = Hyperparameters(name=monitored_metric.name, value=0 if monitored_metric.is_max else 1)
    all_metrics = Hyperparameters(name=f"all {monitored_metric.name}", value=[])

    for sigma in sigma_rule:
        for evaluation_window_size in tqdm(evaluation_window_size_range):
            

            log("Starting new evaluation", f"Evaluation window size {str(evaluation_window_size)}\nSigma rule {str(sigma)}", emphasize=True)
            threshold_main(model_id=model_id, evaluation_window_size=evaluation_window_size, sigma_rule=sigma)
            report = evaluate_main(model_id=model_id, detection_window=evaluation_window_size, folder_message =folder_message , skip_done=False, save = False, d_info=d_info)

        
            if "not applicable" not in str(report['time to detect']):
                times_to_detect.append(report['time to detect'])
                
            new_metric = report[monitored_metric.name]
            all_metrics.value.append(new_metric)
            better = False
            if monitored_metric.is_max:
                if new_metric > current_best.value:
                    current_best.value = new_metric
                    better = True
            else:
                if new_metric < current_best.value:
                    current_best.value = new_metric
                    better = True
            if better:
                best_evaluation_window_size.value = evaluation_window_size
                best_sigma_rule.value = sigma


            output_file = f"{work_dir}/hyperparameter_tuning/sigma_{sigma}_evaluation_window_{evaluation_window_size}_report.json"
            # convert report dictionary to string
            report = json.dumps(report, indent=4, sort_keys=True)
            # save repport to file
            with open(output_file, 'w') as f:
                f.write(report)

    # --------------- Log results -----------------
    log("Hyperparameter tuning", "Evaluation finished")

    for hyperparam in hyperparams:
        log(f"Best {hyperparam.name}", str(hyperparam.value))

    message = "Max " if monitored_metric.is_max else "Min "
    message += current_best.name
    log(message, str(current_best.value))

    # --------------- Save results -----------------
    # combine info to save in a json
    json_dict = {}
    for hyperparam in hyperparams:
        json_dict[hyperparam.name] = hyperparam.value
    json_dict[current_best.name] = current_best.value
    json_dict[f"Average {monitored_metric.name}"] = sum(all_metrics.value)/len(all_metrics.value)
    json_dict[f"Average time to detect"] = sum(times_to_detect)/len(times_to_detect)
    # save json
    with open(f"{work_dir}/hyperparameter_tuning/hyperparameters.json", 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)



    # --------------- Plot histogram of results -----------------
    # plot distribution of times to detect
    sns.histplot(all_metrics.value, kde=True, color='green')
    plt.xlabel(monitored_metric.name)
    plt.ylabel("Density")
    plt.title(f"Distribution of {monitored_metric.name} values")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(f"{work_dir}/hyperparameter_tuning/{monitored_metric.name}_histogram.png")



    
    end = time.time()
    elapsed = end - start
    log("time elapsed in executing hyperparameter search", time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15], time.gmtime(elapsed)))