import numpy as np
import tensorflow as tf

#   Declaring list of features
#   is an array because it can have multiple features
feature_column = [tf.feature_column.numeric_column('x', shape=1)]

#   estimators come in 1k different varieties and are
#   mainly used for training and evaluation
estimator = tf.estimator.LinearRegressor(feature_columns=feature_column)

#   we use numpy here because it is easier to define/manipulate datasets
#   as well as because it is easy to integrate with tensorflow

#   training datasets
x_train = np.array([1, 2, 3, 4])
y_train = np.array([0, -1, -2, -3])

#   evaluation datasets
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7., 0.])

#   Now we need to make the actual functions.
#   num_epochs is how many batches of data we want
#   batch_size is now big each batch should be

#   base function
input_function = tf.estimator.inputs.numpy_input_fn(x={'x': x_train},
                                                    y=y_train,
                                                    batch_size=4,
                                                    num_epochs=None,
                                                    shuffle=True)

train_input_function = tf.estimator.inputs.numpy_input_fn(x={'x': x_train},
                                                          y=y_train,
                                                          batch_size=4,
                                                          num_epochs=1000,
                                                          shuffle=True)

eval_input_function = tf.estimator.inputs.numpy_input_fn(x={'x': x_eval},
                                                         y=y_eval,
                                                         batch_size=4,
                                                         num_epochs=1000,
                                                         shuffle=True)

#   run the base function 1000 times
estimator.train(input_fn=input_function, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_function)
eval_metrics = estimator.evaluate(input_fn=eval_input_function)

print('train metrics: {}'.format(train_metrics))
print('eval metrics: {}'.format(eval_metrics))
