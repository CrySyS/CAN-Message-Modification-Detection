import tensorflow as tf

#helper function for datagenerator
def generate(data, sequence_length, sequence_stride, target=None) :
    return tf.keras.utils.timeseries_dataset_from_array(data, target, sequence_length=sequence_length, sequence_stride=sequence_stride)

#used - this is datagenerator2
#if we want to forecast more than one value, the number of values to forecast is in the 'multistep' parameter, which will create 'multistep'-many target values
#for additional information and example see 'Example - Multistep forecast.ipynb # Példa 2"
def multistep_datagenerator(X_, sample_length, target_length, only_x = False, **kwargs):
 
    # kwargs : for later, if we want to use sampling rate or additional stride, not used at the moment
    
    Y = X_[sample_length:]
    X = X_[:-target_length]

    #these will create non-overlapping targets, no gap between them
    input_dataset = generate(
        data=X, target=None, sequence_length=sample_length, sequence_stride=target_length)
    target_dataset = generate(
        data=Y, target=None, sequence_length=target_length, sequence_stride=target_length)


    # this is the structure:
    #for batch in zip(input_dataset, target_dataset):
    #    inputs, targets = batch

    #for prediction we only want to return samples, without target values
    if only_x:
        print("Datagenerator info: Returning only samples, no targets")
        return input_dataset
    else:
        return tf.data.Dataset.zip((input_dataset, target_dataset))
        