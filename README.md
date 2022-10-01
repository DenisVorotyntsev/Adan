# Tensorflow Adan

Unofficial implementation of Adan optimizer. 

This implementation differs from the official pytorch implementation. 
The main difference is that gradient parameters aren't updated for categorical values which aren't present in the current batch. 
It's especially important for tasks when the batch doesn't contain all possible categorical values. 

See "Test sparse - a lot of categories" in `notebooks/test_adan.ipynb` for illustation.

See the paper for details - [Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models](https://arxiv.org/abs/2208.06677).

See official pytorch implementation - [Adan](https://github.com/sail-sg/Adan).

## Install 

```
pip install adan-tensorflow
```

## Usage example

```
from tf_adan.adan import Adan

model.compile(
    optimizer=Adan(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
```

See `notebooks/example.ipynb` for an example.

## Running tests

To test the correctness of the implementation, 
we're running official pytorch implementation and tensorflow implementation on the same data. 
If the hparams of the optimizers are the same (lr, betas, etc) and initial data is the same, 
loss history and weights after optimization must be the same too.

1. Build docker image 

```
docker build -t latest .
docker run -p 8888:8888 -v $(pwd):/work latest jupyter notebook --ip 0.0.0.0 --port=8888 --allow-root
```

2. Run `notebooks/test_adan.ipynb`

