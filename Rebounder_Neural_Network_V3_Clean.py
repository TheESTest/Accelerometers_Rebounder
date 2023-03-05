# -*- coding: utf-8 -*-
r"""
Elation Sports Technologies LLC
https://www.elationsportstechnologies.com/

Smart Sensing Rebounder Neural Network

pip install tensorflow
(This will install both TensorFlow and Keras.)


"""

import matplotlib.pyplot as plt
import csv,time,pickle

plt.close('all')

# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras import layers
from keras.layers import Dense, SimpleRNN # for creating regular densely-connected NN layers.

import keras.backend as K

# Data manipulation
import pandas as pd # for data manipulation
print('pandas: %s' % pd.__version__) # print version
import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version

# Sklearn
import sklearn # for model evaluation
print('sklearn: %s' % sklearn.__version__) # print version
from sklearn.model_selection import train_test_split # for splitting data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics

# Visualization
import plotly 
import plotly.express as px
import plotly.graph_objects as go
print('plotly: %s' % plotly.__version__) # print version

plot_alpha = 0.25

curr_time_str = time.strftime("%d%b%Y_%H%M%p")

plt.close('all')

# Read in the data csv as a Pandas dataframe
folder_path = r'C:\Your folder path here'

file_names = []
file_names.append(r'Combined_Training_Data')

# Use Combine_CSV_Files_Clean.py to combine the files into Combined_Training_Data.csv
# prior to running this script.

file_type = r'.csv'

#Grab N number of points from each of the 4 x accelerometers to adequately define the shape of an impact
N = 12

# Structure of the data is the following:
# Input layer:
# accelerometer #1 acceleration magnitude readings for N number of points since the start of the current ball impact
# accelerometer #2 acceleration magnitude readings for N number of points since the start of the current ball impact
# accelerometer #3 acceleration magnitude readings for N number of points since the start of the current ball impact
# accelerometer #4 acceleration magnitude readings for N number of points since the start of the current ball impact

# Output layer:
# ball tracking x-position, y-position, radius

numeric_feature_names_x = []
for i in range(1,5):
    for j in range(0,N):
        numeric_feature_names_x.append('d' + str(i) + '_amag_' + str(j))
numeric_feature_names_y = ['c_x','c_y','c_r']

all_dfs = []
all_Xs = []
all_ys = [] 

for i in range(0,len(file_names)):

    file_path = folder_path + '\\' + file_names[i] + file_type
    
    print('Reading dataframe from file: ' + file_names[i] + file_type)
    print()
    
    df=pd.read_csv(file_path, encoding='utf-8')
    
    print(df)
    
    all_dfs.append(df)
    
    numeric_features_x = df[numeric_feature_names_x]
    X = numeric_features_x.values
    
    numeric_features_y = df[numeric_feature_names_y]
    y = numeric_features_y.values
    
    #Center the (x,y) circle tracking data individually for each test run.
    #This is just in case the camera or frame shifted during testing.
    y[:,0] = (y[:,0] - min(y[:,0])) / (max(y[:,0]) - min(y[:,0]))
    y[:,1] = (y[:,1] - min(y[:,1])) / (max(y[:,1]) - min(y[:,1]))
    
    #Zero the X training data by subtracting the initial value. This will
    #account for differences in the tilt angle, etc. at the start of each
    #test run.
    for j in range(0,len(X[0])):
        X[:,j] = X[:,j] - X[0,j]
    
    all_Xs.append(X)
    all_ys.append(y) 

X_combined = all_Xs[0]
y_combined = all_ys[0]
for i in range(1,len(all_Xs)):
    X_combined = np.concatenate((X_combined,all_Xs[i]),axis=0)
    y_combined = np.concatenate((y_combined,all_ys[i]),axis=0)

#Combine all the dataframes into one
df_combined = pd.concat(all_dfs,axis=1)

X = X_combined
y = y_combined

#Optionally shift the webcam tracking data. This is done before the neural
#network training.
#I'm shifting the "y" array, i.e. the solution data used to train the NN.
def center_shift(x,y,bounding_limits_x,bounding_limits_y,center,translation):
    
    #Idenfity all the points to the left, right, above and below the center point.
    indices_left = np.where(x < center[0])[0]
    indices_right = np.where(x >= center[0])[0]
    indices_lower = np.where(y < center[1])[0]
    indices_upper = np.where(y >= center[1])[0]
    
    #Create arrays
    x_shifts = np.zeros(len(x))
    y_shifts = np.zeros(len(y))
    
    x_shifts[indices_left]  = translation[0]  * (x[indices_left] - bounding_limits_x[0]) / (center[0] - bounding_limits_x[0])
    x_shifts[indices_right] = translation[0] * (bounding_limits_x[1] - x[indices_right]) / (bounding_limits_x[1] - center[1])
    y_shifts[indices_lower] = translation[1] * (y[indices_lower] - bounding_limits_y[0]) / (center[1] - bounding_limits_y[0])
    y_shifts[indices_upper] = translation[1] * (bounding_limits_y[1] - y[indices_upper]) / (bounding_limits_y[1] - center[1])
    
    x_new = x + x_shifts
    y_new = y + y_shifts
    
    return x_new, y_new


center_shift_bool = True
if center_shift_bool:
    center = [0.5,0.6]
    translation = [0,-0.1]
    bounding_limits_x = [0,1]
    bounding_limits_y = [0,1]
    
    x_pos_old = np.copy(y[:,0])
    y_pos_old = np.copy(y[:,1])
    center = [0.5,0.6]
    translation = [0,-0.1]
    bounding_limits_x = [0,1]
    bounding_limits_y = [0,1]
    x_pos_new, y_pos_new = center_shift(x_pos_old,y_pos_old,bounding_limits_x,bounding_limits_y,center,translation)
    y[:,0] = x_pos_new
    y[:,1] = y_pos_new
    
    fig,ax = plt.subplots()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bounce Positions Correction Overlay')
    plt.axis('equal')
    plot_alpha = 0.25
    plt.grid(True,alpha=plot_alpha)
    
    #Draw dashed lines from all the points to their new, translated positions
    for i in range(0,len(x_pos_old)):
        plt.plot([x_pos_old[i],x_pos_new[i]],[y_pos_old[i],y_pos_new[i]],'k:',linewidth=1)
    plt.plot(x_pos_old,y_pos_old,'bo',label='Points')
    plt.plot(x_pos_new,y_pos_new,'rs',label='Shifted')
    
    plt.plot([center[0],center[0]+translation[0]],[center[1],center[1]+translation[1]],'k-',linewidth=2)
    plt.plot([center[0]],[center[1]],'og',label='Original Center',markersize=12)
    plt.plot([center[0]+translation[0]],[center[1]+translation[1]],'sc',label='New Center',markersize=12)
    
    #Plot the boundary as well
    boundary_array_x = [bounding_limits_x[0],bounding_limits_x[1],bounding_limits_x[1],bounding_limits_x[0],bounding_limits_x[0]]
    boundary_array_y = [bounding_limits_y[0],bounding_limits_y[0],bounding_limits_y[1],bounding_limits_y[1],bounding_limits_y[0]]
    plt.plot(boundary_array_x,boundary_array_y,'k-')
    plt.legend()
    
    plt.savefig(folder_path + '\\' + 'Center_Shift_Overlay' + '_' + curr_time_str + '.png', dpi = 200)
    
    #Let's plot the points before and after the shift
    fig,ax = plt.subplots()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Webcam-Tracked Bounce Positions\nBefore Correction')
    plt.axis('equal')
    plot_alpha = 0.25
    plt.grid(True,alpha=plot_alpha)
    plt.plot(x_pos_old,y_pos_old,'bo',label='Uncorrected')
    plt.plot([center[0]],[center[1]],'oc',label='Original Center (0.5,0.6)',markersize=8)
    boundary_array_x = [bounding_limits_x[0],bounding_limits_x[1],bounding_limits_x[1],bounding_limits_x[0],bounding_limits_x[0]]
    boundary_array_y = [bounding_limits_y[0],bounding_limits_y[0],bounding_limits_y[1],bounding_limits_y[1],bounding_limits_y[0]]
    plt.plot(boundary_array_x,boundary_array_y,'k-')
    plt.legend()
    plt.savefig(folder_path + '\\' + 'Bounce_Positions_Before' + '_' + curr_time_str + '.png', dpi = 200)
    
    fig,ax = plt.subplots()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Webcam-Tracked Bounce Positions\nAfter Correction')
    plt.axis('equal')
    plot_alpha = 0.25
    plt.grid(True,alpha=plot_alpha)
    plt.plot(x_pos_new,y_pos_new,'ro',label='Corrected')
    plt.plot([center[0]+translation[0]],[center[1]+translation[1]],'o',color='orange',label='New Center (0.5,0.5)',markersize=8)
    boundary_array_x = [bounding_limits_x[0],bounding_limits_x[1],bounding_limits_x[1],bounding_limits_x[0],bounding_limits_x[0]]
    boundary_array_y = [bounding_limits_y[0],bounding_limits_y[0],bounding_limits_y[1],bounding_limits_y[1],bounding_limits_y[0]]
    plt.plot(boundary_array_x,boundary_array_y,'k-')
    plt.legend()
    plt.savefig(folder_path + '\\' + 'Bounce_Positions_After' + '_' + curr_time_str + '.png', dpi = 200)
    
    print('Center shift applied.')

else:
    print('No center shift applied.')


#Screen the training data for outliers
bad_indices = []
#Set limits for screening to, for example, +/- 3-sigma (should encapsulate 99.7% of the samples)
sigma_limit_count = 3

for i in range(0,len(X[0])):
    avg_curr = np.mean(X[:,i])
    std_curr = np.std(X[:,i])
    limit_lower = avg_curr - sigma_limit_count * std_curr
    limit_upper = avg_curr + sigma_limit_count * std_curr
    bad_indices_curr_1 = np.where(X[:,i] < limit_lower)
    bad_indices_curr_2 = np.where(X[:,i] > limit_upper)
    if len(bad_indices_curr_1[0]) > 0:
        for ind in bad_indices_curr_1[0]:
            bad_indices.append(ind)
    if len(bad_indices_curr_2[0]) > 0:
        for ind in bad_indices_curr_2[0]:
            bad_indices.append(ind)


for i in range(0,len(y[0])):
    avg_curr = np.mean(y[:,i])
    std_curr = np.std(y[:,i])
    limit_lower = avg_curr - sigma_limit_count * std_curr
    limit_upper = avg_curr + sigma_limit_count * std_curr
    bad_indices_curr_1 = np.where(y[:,i] < limit_lower)
    bad_indices_curr_2 = np.where(y[:,i] > limit_upper)
    if len(bad_indices_curr_1[0]) > 0:
        for ind in bad_indices_curr_1[0]:
            bad_indices.append(ind)
    if len(bad_indices_curr_2[0]) > 0:
        for ind in bad_indices_curr_2[0]:
            bad_indices.append(ind)

bad_indices = np.unique(bad_indices)
X = np.delete(X,bad_indices,0)
y = np.delete(y,bad_indices,0)

#All the X (training) and y (output) data from all the training session files
#is now combined into a big list. Normalize that big list with respect to itself.
max_min_vals = []
normalize_the_data_bool = False
for i in range(0,len(X[0])):
    max_min_vals.append([np.min(X[:,i]), np.max(X[:,i])])
    if normalize_the_data_bool:
        X[:,i] = (X[:,i] - np.min(X[:,i]))/(np.max(X[:,i]) - np.min(X[:,i]))

#Save the max and min training values to a CSV file.
with open(folder_path + '\\' + 'Training_Max_Min_Vals' + '.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for i in range(0,len(max_min_vals)):
        spamwriter.writerow(max_min_vals[i])

if normalize_the_data_bool:
    for i in range(0,len(y[0])):
        y[:,i] = (y[:,i] - np.min(y[:,i]))/(np.max(y[:,i]) - np.min(y[:,i]))

#Try removing the circle radius from the test data, which might make it
#easier to determine the landing position.
remove_radius_bool = True
if remove_radius_bool: y = y[:,0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Define the loss function as the Euclidian distance.
#https://stackoverflow.com/questions/59659494/euclidean-distance-in-keras
def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


#https://stackoverflow.com/questions/51683915/how-can-i-limit-regression-output-between-0-to-1-in-keras
#Want sigmoid for the output layer if you want that output to range from
#0 to 1, or, set it to tanh if you want it to range from -1 to 1

model = Sequential(name="Model") # Model
model.add(Input(shape=(len(X[0]),), name='Input-Layer')) # Input Layer - need to speicfy the shape of inputs
model.add(Dense(N*4, activation='sigmoid', name='Hidden-Layer1')) # Hidden Layer
model.add(Dense(N*4, activation='sigmoid', name='Hidden-Layer2')) # Hidden Layer
model.add(Dense(len(y[0]), activation='sigmoid', name='Output-Layer')) # Output Layer, sigmoid(x) = 1 / (1 + exp(-x))


#https://www.tensorflow.org/guide/keras/rnn
try_RNN_bool = True
if try_RNN_bool:
    #The SimpleRNN layer expects the input to have 3 dimensions, but you're providing an input with 2 dimensions.
    #To fix this issue, you need to reshape your input data to have 3 dimensions.
    #The first dimension represents the number of samples, the second dimension represents the number of
    #timesteps, and the third dimension represents the number of features in each timestep.
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    model = Sequential()
    model.add(SimpleRNN(N*4, input_shape=(None, N*4)))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(len(y[0]), activation='sigmoid'))


training_metric_str = 'Accuracy'

##### Step 4 - Compile keras model
model.compile(optimizer='adam', # default='rmsprop', an algorithm to be used in backpropagation
              loss=euclidean_distance_loss, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
              metrics=[training_metric_str], # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics. Metric instance. 
              loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
              weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
              run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
              steps_per_execution=1 # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
             )

num_epochs = 5000

##### Step 5 - Fit keras model on the dataset
history = model.fit(X_train, # input data
          y_train, # target data
          batch_size=16, # Number of samples per gradient update. If unspecified, batch_size will default to 32.
          epochs=num_epochs, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
          verbose='auto', # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
          callbacks=None, # default=None, list of callbacks to apply during training. See tf.keras.callbacks
          validation_split=0.0, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. 
          #validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch. 
          shuffle=True, # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
          class_weight=None, # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
          sample_weight=None, # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
          initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
          steps_per_epoch=None, # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. 
          validation_steps=None, # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
          validation_batch_size=None, # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
          validation_freq=1, # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
          max_queue_size=10, # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
          workers=1, # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
          use_multiprocessing=False, # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. 
         )

#View the change in the training metric over the course of training
print()
print('History keys:')
print(history.history.keys())

fig,ax = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Metric Evolution: ' + training_metric_str)
plt.grid(True,alpha=plot_alpha)
plt.plot(history.history[training_metric_str],'-',color='C0',label=training_metric_str)
plt.legend()
plt.savefig(folder_path + '\\' + 'Training-Evolution' + '_' + curr_time_str + '.png', dpi = 200)

fig,ax = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Loss Evolution During Training')
plt.grid(True,alpha=plot_alpha)
plt.plot(history.history['loss'],'-',color='C1',label='Loss')
plt.legend()
plt.savefig(folder_path + '\\' + 'Training-Evolution-Loss' + '_' + curr_time_str + '.png', dpi = 200)


#With the neural network trained, try making some predictions using the test data
#test_output = model.predict(X_test)
output_guesses = model.predict(X_train)
errors_raw = output_guesses - y_train

#Convert the x and y position errors into a single mean squared position error
#errors list is [pos,radius,x,y], or just [distance,x,y]
errors = []
errors.append(np.sqrt(np.power(errors_raw[:,0],2) + np.power(errors_raw[:,1],2)))
#errors.append(errors_raw[:,2])
errors.append(errors_raw[:,0])
errors.append(errors_raw[:,1])

error_avgs = []
error_stds = []
for i in range(0,len(errors)):
    error_avgs.append(np.mean(errors[i]))
    error_stds.append(np.std(errors[i]))

print()
print('error_avgs: ' + str(error_avgs))
print('error_stds: ' + str(error_stds))

from scipy.stats import norm
from scipy.stats import lognorm

#Make a histogram of the errors
fig,ax = plt.subplots()
the_hist = plt.hist(errors[0], bins = 100, alpha = 0.5, label = 'Position Error')
plt.grid(True,alpha = plot_alpha)
plt.title('Neural Network Output Error Histogram')
plt.xlabel('Value')
plt.ylabel('Count')

# Fit a lognormal distribution to the data:
data = errors[0]
shape, loc, scale = lognorm.fit(data, floc=0)
# Calculate the x and y values for the lognormal distribution curve
x = np.linspace(data.min(), data.max(), 1000)
y = lognorm.pdf(x, shape, loc=0, scale=scale) * len(data) * np.diff(the_hist[1])[0]
# Plot the lognormal distribution curve
plt.plot(x, y, 'r', linewidth=2, label='Lognormal Distribution')

# Calculate the lognormal confidence interval
#1.96 std devs corresponds to 95% confidence interval
#2.58 std devs corresponds to 99% confidence interval
log_data = np.log(data)
log_mean = np.mean(log_data)
log_std = np.std(log_data)
lower_limit = np.exp(log_mean - 1.96*log_std)
upper_limit = np.exp(log_mean + 1.96*log_std)
lower_limit_99 = np.exp(log_mean - 2.58*log_std)
upper_limit_99 = np.exp(log_mean + 2.58*log_std)
geometric_mean = np.exp(log_mean)

# Add the confidence interval to the plot
plt.axvline(upper_limit, color='r', linestyle='--', linewidth=2, label='95% Confidence Interval')
plt.axvline(upper_limit_99, color='k', linestyle='--', linewidth=2, label='99% Confidence Interval')

plt.legend()
plt.savefig(folder_path + '\\' + 'Histogram' + '_' + curr_time_str + '.png', dpi = 200)
pickle.dump(fig, open(folder_path + '\\' + 'Histogram' + '_' + curr_time_str + '.pkl', 'wb'))

print()
print('Lognormal 95% Confidence Integral Limits: ' + str(lower_limit) + ',' + str(upper_limit))
print('Geomtric mean of the data: ' + str(geometric_mean))

#Because the position error can't be less than zero, take the log of the
#position error from the neural net output (i.e. log-normal distribution).
pos_error_log = np.log(errors[0])
error_log_avg = np.mean(pos_error_log)
error_log_std = np.std(pos_error_log)

error_exp_recalc_avg = np.mean(np.exp(error_log_avg))
error_exp_recalc_std = np.std(np.exp(error_log_std))


fig,ax = plt.subplots()
the_hist = plt.hist(pos_error_log, bins = 30, alpha = 0.5, color='g', label = 'Log of Position Error')
plt.grid(True,alpha = plot_alpha)
plt.title('Log of Neural Network Output Error Histogram')
plt.xlabel('Value')
plt.ylabel('Count')
# Fit a normal distribution to the data:
data = pos_error_log
mu, std = norm.fit(pos_error_log) 
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu, std) * len(data) * np.diff(the_hist[1])[0]
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.legend()
plt.savefig(folder_path + '\\' + 'Histogram-Log' + '_' + curr_time_str + '.png', dpi = 200)
pickle.dump(fig, open(folder_path + '\\' + 'Histogram-Log' + '_' + curr_time_str + '.pkl', 'wb'))

#Plot the distance error as a map
fig,ax = plt.subplots()
plt.grid(True,alpha=0.5)
plt.xlabel('X [normalized]')
plt.ylabel('Y [normalized]')
plt.title('Map of Predicted Versus Actual Ball Landing Positions')
plt.axis('equal')

plt.plot(output_guesses[:,0],output_guesses[:,1],'rs',label='NN Output')
plt.plot(y_train[:,0],y_train[:,1],'bo',label='Actual')

for i in range(0,len(y_train)):
    testPosition_curr = y_train[i]
    output_curr = output_guesses[i]
    
    x1 = output_curr[0]
    x2 = testPosition_curr[0]
    y1 = output_curr[1]
    y2 = testPosition_curr[1]
    
    plt.plot([x1,x2],[y1,y2],'k--')

plt.plot([0,1,1,0,0],[0,0,1,1,0],'k--')
plt.legend()
plt.savefig(folder_path + '\\' + 'MapOfPredictedVersusActualLandingPosition' + '_' + curr_time_str + '.png', dpi = 200)
pickle.dump(fig, open(folder_path + '\\' + 'MapOfPredictedVersusActualLandingPosition' + '_' + curr_time_str + '.pkl', 'wb'))


print()
print('Saving model and its lite version...')

#Save the trained model as the filetype ".pb" along with some folders of
#other assets for the model, if applicable.
#https://www.tensorflow.org/guide/saved_model
#The saved model can be loaded using tf.saved_model.load(file_path) --> Except...! Keras has to use a slightly different function to load a model.
model_save_folder = folder_path + '\\' + 'Model_' + curr_time_str + '\\'
tf.saved_model.save(model, model_save_folder)

#Also save using the Keras package to be safe
#https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model
model_keras_save_folder = folder_path + '\\' + 'Keras_Model_' + curr_time_str + '\\'
tf.keras.models.save_model(model, model_keras_save_folder)



