# Estimator
## A predicatable way to train your deep learning model

Training deep learning model in tensorflow should be easy and fast. This library introduce a predictable object oriented approach to archieve booth simplicity and granulatiry while preserving customisation.

The aim of this library is to allows the developer to just define a model and an input pipeline in order to train, evaluate and test their model without writing any train loop and reinvent the wheel everytime.


### Installation

The package is available on pip:

```
pip install TODO
```

or you can clone this directory and manually place the `estimator` folder in your project.

Then it can be imported in your script

```python
from estimator.Estimator import Estimator
```

### Motivation
TensorFlow provides an [Estimator](METTI LINK) implementation that is more than a black box that a library. It cannot be easily extended and customise for this reasons I decided to implemented my owm.

### Example

It follows a very simple example that will use some random data and a basic feed forward network

```python
from estimator.Estimator import Estimator, Mode
import numpy as np
import tensorflow as tf

EPOCHS = 100
# create the dataset
train_data = (np.random.rand(1000,2),np.random.rand(1000,1))
test_data = (np.random.rand(100,2),np.random.rand(100,1))
# define the input function
input_fn = Estimator.create_input_fn(input_shape=[None,2],output_shape=[None,1])
# define the model builder
def model_builder(x, y, config):
    # config is a dictionary that can be passed to the estimator
    net = tf.layers.dense(x, 16, activation=tf.nn.relu)
    predictions = tf.layers.dense(net, 1, activation=tf.nn.sigmoid)

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    # it must return a dictionary contain the operation to train, predict and evaluate
    return {
            Mode.TRAIN: {'train_step': train_step },
            Mode.PREDICT: { 'predictions': predictions },
            Mode.EVAL: { 'loss': loss } # used as metrics
    }


estimator = Estimator(model_builder, input_fn)
# we can define a batch size before train, default is one
estimator.train(data=train_data, epochs=EPOCHS, batch_size=64)
res = estimator.evaluate(data=test_data)
print(res)

pred = estimator.predict(np.array([[2,1]]))
print(pred)

```
### Data creation

In this example we fist create two random dataset using `numpy`

```python
train_data = (np.random.rand(1000,2),np.random.rand(1000,1))
test_data = (np.random.rand(100,2),np.random.rand(100,1))
```
### Input pipeline
Then we define an `input_fn` by calling an utily function from the `Estimator` class that takes the input and output shape.
 
```python
input_fn = Estimator.create_input_fn(input_shape=[None,2],output_shape=[None,1])
```
You must manually specify the shapes since, internally, the instimator will create a generic iterator that will switch between the train Dataset and the eval Dataset. 
You must pass `None` as first dimension since it will be for batching.


### Model creation

```python
def model_builder(x, y, config):
    # config is a dictionary that can be passed to the estimator
    net = tf.layers.dense(x, 16, activation=tf.nn.relu)
    predictions = tf.layers.dense(net, 1, activation=tf.nn.sigmoid)

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    # it must return a dictionary contain the operation to train, predict and evaluate
    return {
            Mode.TRAIN: {'train_step': train_step },
            Mode.PREDICT: { 'predictions': predictions },
            Mode.EVAL: { 'loss': loss } # used as metrics
    }
```
The Estimator needs a function that return a dictionary with the operations to run for each mode. The modes are three:

* Mode.Train: defined the operations that will be run when we call the `.train` method
* Mode.Predict: defined the operations that will be run when we call the `.predict` method
* Mode.Eval: defined the operations that will be run when we call the `.eval` method. They are our metrics.

### Estimator creation

Create the estimator is super easy

```python
estimator = Estimator(model_builder, input_fn)
```

### Train

To train, just call the `.train` method with the train data and a batch size as input:

```python
estimator.train(data=train_data, epochs=EPOCHS, batch_size=64)
```

### Evaluate

To evaluate, similar to before, call the `.evaluate` method. You can also pass a batch size, the **default is the size of the data input**

```python
res = estimator.evaluate(data=test_data)
```
It will return a dictionary that contains the results of all the operation defined in the field `Mode.EVAL` of the dictionary that `model_builder` returns. In our case, the output is:

```
{'loss': [0.08878718]}
```

### Predict
To predict, call the `.predict` method with an input. 

```python
pred = estimator.predict(np.array([[2,1]]))
print(pred)
```

It will return a dictionary that contains the results of all the operation defined in the field `Mode.PREDICT` of the dictionary that `model_builder` returns. In our case, the output is:

```python
{'predictions': array([[0.61255413]], dtype=float32)}
```