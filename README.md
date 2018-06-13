# Estimator
## A predictable way to train your deep learning model

Training deep learning model in tensorflow should be easy and fast. This library introduces a predictable object-oriented approach to achieve booth simplicity and granularity while preserving customization.

The aim of this library is to allows the developer to just define a model and an input pipeline in order to train, evaluate and test their model without writing any train loop and reinvent the wheel every time.

### Installation

** At the moment the package is NOT available on pip **
<!-- The package is available on pip: -->
You can clone the repo and install it directly from it

```
git clone THIS_REPO
cd THIS_REPO/estimator
pip install .
```

Then it can be imported in your script

```python
from estimator.Estimator import Estimator
```

### Motivation
TensorFlow provides an [Estimator](https://www.tensorflow.org/programmers_guide/estimators) implementation that is more a black box than a library. It cannot be easily extended and customize for this reasons I decided to implement my own.
### Example
You can also check out the this [notebook]().

It follows a very simple example in which we create a basic feed forward neural network and we train it with random numbers

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
You must manually specify the shapes since, internally, the Estimator will create a generic iterator that will switch between the train Dataset and the eval Dataset. 
You must pass `None` as the first dimension since the first dimension is used later for batching.

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
The Estimator needs a function that returns a dictionary with the operations to run for each mode. The modes are three:

* Mode.Train: defines the operations that will be run when we call the `.train` method
* Mode.Predict: defines the operations that will be run when we call the `.predict` method
* Mode.Eval: defines the operations that will be run when we call the `.eval` method. These are your metrics. For example, accuracy

**The model is built only once.**

### Estimator creation

Create an estimator is super easy

```python
estimator = Estimator(model_builder, input_fn)
```

### Train

To train, just call the `.train` method with the train data and a batch size as input:

```python
estimator.train(data=train_data, epochs=EPOCHS, batch_size=64)
```

### Evaluate

To evaluate, call the `.evaluate` method. You can also pass a batch size, the **default is the size of the data input**

```python
res = estimator.evaluate(data=test_data)
```
It will return a dictionary that contains the results of all the operation defined in the field `Mode.EVAL` of the dictionary that `model_builder` returns. In our case, the output is:

```
{'loss': [0.08878718]}
```

### Train and Evaluate

You can train and evaluate by calling

```python
estimator.train_and_evaluate(data=train_data, epoches=10, validation=val_data, batch_size=64, batch_size_eval=32, every=2)

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
## Customisation

Hooks can be attach the Estimator in order to decorate it with new functionalities. Each hook must subclass the `Hook` class.

```
class Hook:
    def before_build_model(self, estimator):
        pass

    def after_build_model(self, estimator):
        pass

    def before_run_epoch(self, estimator, epoch, data, batch_n, tot_res):
        pass

    def after_run_epoch(self, estimator, epoch, data, batch_n, tot_res):
        pass

    def before_run_batch(self, estimator, res, i):
        pass

    def after_run_batch(self, estimator, res, i, tot_res):
        pass
```

This class exposes a wide range of function that will be called by the estimator. For example, a very simple logger can be defined as follow:

```python
class Logger(Hook):

    def after_run_epoch(self, estimator, epoch, data, batch_n, tot_res):
        # tot_res is an array with the result for each batch, in this case { 'loss' : [...] }
        # take the mean of each metric key
        mean_res = {k: np.mean(tot_res[k]) for k in estimator.metrics[Mode.EVAL].keys()}
        print(mean_res)
```

Then it can be passed to the Estimator as argument

```python
estimator = Estimator(model_builder, input_fn, hooks=[Logger()])
# we can define a batch size before train, default is one
estimator.train(data=train_data, epochs=EPOCHS, batch_size=64)
res = estimator.evaluate(data=test_data)
pred = estimator.predict(np.array([[2,1]]))
print(pred)
```
The output is:

```python
...
{'loss': 0.08635884}
{'loss': 0.084242225}
{'loss': 0.08413396}
{'loss': 0.082261935}
{'loss': 0.08626987}
{'loss': 0.08318842}
{'loss': 0.08242814}
{'loss': 0.08449897}
{'loss': 0.08369716}
{'loss': 0.0927686}
{'predictions': array([[0.45650724]], dtype=float32)}
```

A full example can be found in `example/hook/main.py`




## Saving and Restoring
TODO

## TODO
* Explain how Hooks work
* Write doc
