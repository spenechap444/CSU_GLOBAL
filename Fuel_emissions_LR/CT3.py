import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import tensorflow_docs as tfdocs

dataset_path = tf.keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,na_values = "?", comment='\t',sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset.tail()
train_dataset = dataset.sample(frac=0.8,random_state=0)  
test_dataset = dataset.drop(train_dataset.index) 
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#z-score normalization
def norm(x):    
    return (x - train_stats['mean']) / train_stats['std']  

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#Additional step to remove any NA values, otherwise the calculations for MAE & MSE will be nan
normed_train_data.fillna(normed_train_data.mean(), inplace=True)
normed_test_data.fillna(normed_test_data.mean(), inplace=True)


# train_labels.head()
len(train_dataset.keys())
def build_model():    
    model = tf.keras.Sequential([tf.keras.layers.Dense(64, 
                                           activation='relu', 
                                           input_shape=[len(train_dataset.keys())], # 7 features
                                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                                tf.keras.layers.Dense(64, 
                                           activation='relu'),      
                                tf.keras.layers.Dense(1)])      
    optimizer = tf.keras.optimizers.RMSprop(0.001)      
    model.compile(loss='mse', 
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])    
    return model
model = build_model()

EPOCHS = 1000
history = model.fit(normed_train_data, 
                    train_labels,    
                    epochs=EPOCHS,
                    validation_split = 0.2, 
                    verbose=0)    
hist = pd.DataFrame(history.history)  
hist['epoch'] = history.epoch  
hist.tail()
