"""

发布于 https://raw.githubusercontent.com/keras-team/keras-io/master/examples/vision/forwardforward.py

Title: Using the Forward-Forward Algorithm for Image Classification
Author: [Suvaditya Mukherjee](https://twitter.com/halcyonrayes)
Date created: 2023/01/08
Last modified: 2023/01/08
Description: Training a Dense-layer model using the Forward-Forward algorithm.
Accelerator: GPU
"""

"""
## 概述

以下示例探讨了如何使用前向前向算法来进行训练，而不是传统的反向传播方法，正如 [The Forward-Forward Algorithm: Some Preliminary Investigations(2022)](https://www.cs.toronto.edu/~hinton/FFA13.pdf)中提出的一样。


The concept was inspired by the understanding behind
[Boltzmann Machines](http://www.cs.toronto.edu/~fritz/absps/dbm.pdf). Backpropagation
involves calculating the difference between actual and predicted output via a cost
function to adjust network weights. On the other hand, the FF Algorithm suggests the
analogy of neurons which get "excited" based on looking at a certain recognized
combination of an image and its correct corresponding label.

该概念灵感来源于[Boltzmann Machines](http://www.cs.toronto.edu/~fritz/absps/dbm.pdf)背后的理解。反向传播涉及通过成本函数计算实际输出和预测输出之间的差异以调整网络权重。另一方面，FF算法则提出了神经元的类比，即神经元根据观察图像的某个特定识别组合及其正确的对应标签而被“激活”。

This method takes certain inspiration from the biological learning process that occurs in
the cortex. A significant advantage that this method brings is the fact that
backpropagation through the network does not need to be performed anymore, and that
weight updates are local to the layer itself.

此方法从皮层中发生的生物学习过程中获得了灵感。该方法带来的一个重要优势是不再需要进行网络反向传播，且权重更新是局部的。

As this is yet still an experimental method, it does not yield state-of-the-art results.
But with proper tuning, it is supposed to come close to the same.
Through this example, we will examine a process that allows us to implement the
Forward-Forward algorithm within the layers themselves, instead of the traditional method
of relying on the global loss functions and optimizers.

由于这仍然是一种实验性方法，它并没有产生最先进的结果。
但是通过适当的调整，它应该可以接近同样的效果。
通过这个例子，我们将研究一种允许我们在层内部实现Forward-Forward算法的过程，而不是依赖于全局损失函数和优化器的传统方法。

The tutorial is structured as follows:

- Perform necessary imports
- Load the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- Visualize Random samples from the MNIST dataset
- Define a `FFDense` Layer to override `call` and implement a custom `forwardforward`
method which performs weight updates.
- Define a `FFNetwork` Layer to override `train_step`, `predict` and implement 2 custom
functions for per-sample prediction and overlaying labels
- Convert MNIST from `NumPy` arrays to `tf.data.Dataset`
- Fit the network
- Visualize results
- Perform inference on test samples

As this example requires the customization of certain core functions with
`keras.layers.Layer` and `keras.models.Model`, refer to the following resources for
a primer on how to do so:

- [Customizing what happens in `model.fit()`](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
- [Making new Layers and Models via subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
本教程的结构如下：

- 执行必要的导入操作
- 加载 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)
- 可视化来自 MNIST 数据集的随机样本
- 定义 `FFDense` 层来重写 `call` 方法并实现自定义的 `forward` 方法，该方法执行权重更新。
- 定义 `FFNetwork` 层来重写 `train_step`、`predict` 方法并实现 2 个自定义函数，用于每个样本的预测和叠加标签
- 将 MNIST 从 `NumPy` 数组转换为 `tf.data.Dataset`
- 训练网络
- 可视化结果
- 在测试样本上执行推断

由于此示例需要使用 `keras.layers.Layer` 和 `keras.models.Model` 自定义某些核心函数，请参考以下资源，了解如何进行自定义：

- [自定义 `model.fit()` 中发生的事情](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
- [通过子类化创建新的层和模型](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
"""

"""
## 设置import
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
from tensorflow.compiler.tf2xla.python import xla

"""
## Load the dataset and visualize the data

We use the `keras.datasets.mnist.load_data()` utility to directly pull the MNIST dataset
in the form of `NumPy` arrays. We then arrange it in the form of the train and test
splits.

Following loading the dataset, we select 4 random samples from within the training set
and visualize them using `matplotlib.pyplot`.

## 加载数据集并可视化数据

我们使用 `keras.datasets.mnist.load_data()` 工具直接获取 MNIST 数据集，以 `NumPy` 数组的形式呈现。然后我们将其按照训练集和测试集的形式进行排列。

在加载数据集后，我们从训练集中选择 4 个随机样本，并使用 `matplotlib.pyplot` 将它们可视化。
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("4 Random Training samples and labels")
idx1, idx2, idx3, idx4 = random.sample(range(0, x_train.shape[0]), 4)

img1 = (x_train[idx1], y_train[idx1])
img2 = (x_train[idx2], y_train[idx2])
img3 = (x_train[idx3], y_train[idx3])
img4 = (x_train[idx4], y_train[idx4])

imgs = [img1, img2, img3, img4]

plt.figure(figsize=(10, 10))

for idx, item in enumerate(imgs):
    image, label = item[0], item[1]
    plt.subplot(2, 2, idx + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"Label : {label}")
plt.show()

"""
## Define `FFDense` custom layer

In this custom layer, we have a base `keras.layers.Dense` object which acts as the
base `Dense` layer within. Since weight updates will happen within the layer itself, we
add an `keras.optimizers.Optimizer` object that is accepted from the user. Here, we
use `Adam` as our optimizer with a rather higher learning rate of `0.03`.

Following the algorithm's specifics, we must set a `threshold` parameter that will be
used to make the positive-negative decision in each prediction. This is set to a default
of 2.0.
As the epochs are localized to the layer itself, we also set a `num_epochs` parameter
(defaults to 50).

We override the `call` method in order to perform a normalization over the complete
input space followed by running it through the base `Dense` layer as would happen in a
normal `Dense` layer call.

We implement the Forward-Forward algorithm which accepts 2 kinds of input tensors, each
representing the positive and negative samples respectively. We write a custom training
loop here with the use of `tf.GradientTape()`, within which we calculate a loss per
sample by taking the distance of the prediction from the threshold to understand the
error and taking its mean to get a `mean_loss` metric.

With the help of `tf.GradientTape()` we calculate the gradient updates for the trainable
base `Dense` layer and apply them using the layer's local optimizer.

Finally, we return the `call` result as the `Dense` results of the positive and negative
samples while also returning the last `mean_loss` metric and all the loss values over a
certain all-epoch run.

## 定义 `FFDense` 自定义层

在这个自定义层中，我们有一个基础的 `keras.layers.Dense` 对象，作为内部的基础 `Dense` 层。由于权重更新将在层本身内部发生，我们添加了一个可由用户接受的 `keras.optimizers.Optimizer` 对象。这里，我们使用 `Adam` 作为优化器，学习率设置为较高的 `0.03`。

根据算法的特定要求，我们必须设置一个 `threshold` 参数，用于在每个预测中进行正负决策。默认设置为 2.0。由于这个层的 epochs 局限于层本身，我们还设置了一个 `num_epochs` 参数（默认为 50）。

我们重写了 `call` 方法，以便在完成对整个输入空间的归一化后，将其通过基础 `Dense` 层运行，就像在正常的 `Dense` 层调用中一样。

我们实现了前向前向（Forward-Forward）算法，它接受两种类型的输入张量，分别表示正样本和负样本。我们在这里编写了一个自定义训练循环，使用 `tf.GradientTape()`，在其中通过将预测与阈值之间的距离来计算每个样本的损失，以理解错误，并取其平均值得到 `mean_loss` 指标。

使用 `tf.GradientTape()`，我们计算可训练的基础 `Dense` 层的梯度更新，并使用该层的本地优化器进行应用。

最后，我们返回 `call` 结果作为正负样本的 `Dense` 结果，同时返回最后的 `mean_loss` 指标和某个所有 epoch 运行期间的所有损失值。
"""


class FFDense(keras.layers.Layer):
    """
    A custom ForwardForward-enabled Dense layer. It has an implementation of the
    Forward-Forward network internally for use.
    This layer must be used in conjunction with the `FFNetwork` model.
    一个自定义的 ForwardForward 启用的 Dense 层。它内部实现了 Forward-Forward 网络以供使用。这个层必须与 `FFNetwork` 模型一起使用。
    """

    def __init__(
        self,
        units,
        optimizer,
        loss_metric,
        num_epochs=50,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.relu = keras.layers.ReLU()
        self.optimizer = optimizer
        self.loss_metric = loss_metric
        self.threshold = 1.5
        self.num_epochs = num_epochs

    # We perform a normalization step before we run the input through the Dense
    # layer.
    # 在将输入传递给 Dense 层之前，我们执行一个归一化步骤。

    def call(self, x):
        x_norm = tf.norm(x, ord=2, axis=1, keepdims=True)
        x_norm = x_norm + 1e-4
        x_dir = x / x_norm
        res = self.dense(x_dir)
        return self.relu(res)

    # The Forward-Forward algorithm is below. We first perform the Dense-layer
    # operation and then get a Mean Square value for all positive and negative
    # samples respectively.
    # The custom loss function finds the distance between the Mean-squared
    # result and the threshold value we set (a hyperparameter) that will define
    # whether the prediction is positive or negative in nature. Once the loss is
    # calculated, we get a mean across the entire batch combined and perform a
    # gradient calculation and optimization step. This does not technically
    # qualify as backpropagation since there is no gradient being
    # sent to any previous layer and is completely local in nature.
    """
    以下是 Forward-Forward 算法。我们首先执行 Dense 层操作，然后分别对所有正样本和负样本获取均方误差值。自定义损失函数找到均方结果与我们设置的阈值值（超参数）之间的距离，这将定义预测的性质（是正还是负）。计算损失后，我们对整个批次进行平均值计算，并执行梯度计算和优化步骤。这不是严格意义上的反向传播，因为没有梯度被发送到任何先前的层，完全是本地的过程。
    """

    def forward_forward(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            with tf.GradientTape() as tape:
                g_pos = tf.math.reduce_mean(tf.math.pow(self.call(x_pos), 2), 1)
                g_neg = tf.math.reduce_mean(tf.math.pow(self.call(x_neg), 2), 1)

                loss = tf.math.log(
                    1
                    + tf.math.exp(
                        tf.concat([-g_pos + self.threshold, g_neg - self.threshold], 0)
                    )
                )
                mean_loss = tf.cast(tf.math.reduce_mean(loss), tf.float32)
                self.loss_metric.update_state([mean_loss])
            gradients = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.dense.trainable_weights))
        return (
            tf.stop_gradient(self.call(x_pos)),
            tf.stop_gradient(self.call(x_neg)),
            self.loss_metric.result(),
        )


"""
## Define the `FFNetwork` Custom Model

With our custom layer defined, we also need to override the `train_step` method and
define a custom `keras.models.Model` that works with our `FFDense` layer.

For this algorithm, we must 'embed' the labels onto the original image. To do so, we
exploit the structure of MNIST images where the top-left 10 pixels are always zeros. We
use that as a label space in order to visually one-hot-encode the labels within the image
itself. This action is performed by the `overlay_y_on_x` function.

We break down the prediction function with a per-sample prediction function which is then
called over the entire test set by the overriden `predict()` function. The prediction is
performed here with the help of measuring the `excitation` of the neurons per layer for
each image. This is then summed over all layers to calculate a network-wide 'goodness
score'. The label with the highest 'goodness score' is then chosen as the sample
prediction.

The `train_step` function is overriden to act as the main controlling loop for running
training on each layer as per the number of epochs per layer.
## 定义 `FFNetwork` 自定义模型

定义了我们的自定义层后，我们还需要重写 `train_step` 方法，并定义一个与我们的 `FFDense` 层配合使用的自定义 `keras.models.Model`。

对于这个算法，我们必须将标签“嵌入”到原始图像中。为此，我们利用 MNIST 图像的结构，其中左上角的 10 个像素始终为零。我们将其用作标签空间，以便在图像本身内部对标签进行可视化的独热编码。这个操作是由 `overlay_y_on_x` 函数执行的。

我们使用一个每个样本预测函数来分解预测函数，然后由重写的 `predict()` 函数对整个测试集进行调用。在这里，通过测量每个图像的每个层的神经元的“激活”来进行预测。然后将其在所有层上相加，以计算整个网络的“好度分数”。具有最高“好度分数”的标签被选择为样本预测。

`train_step` 函数被重写为作为在每个层上按照每层的 epochs 数量运行训练的主要控制循环。
"""


class FFNetwork(keras.Model):
    """
    A `keras.Model` that supports a `FFDense` network creation. This model
    can work for any kind of classification task. It has an internal
    implementation with some details specific to the MNIST dataset which can be
    changed as per the use-case.
    一个支持 `FFDense` 网络创建的 `keras.Model`。这个模型可以适用于任何类型的分类任务。它有一个内部实现，其中一些细节针对于 MNIST 数据集，可以根据具体用例进行更改。
    """

    # Since each layer runs gradient-calculation and optimization locally, each
    # layer has its own optimizer that we pass. As a standard choice, we pass
    # the `Adam` optimizer with a default learning rate of 0.03 as that was
    # found to be the best rate after experimentation.
    # Loss is tracked using `loss_var` and `loss_count` variables.
    # Use legacy optimizer for Layer Optimizer to fix issue
    # https://github.com/keras-team/keras-io/issues/1241
    """
    由于每个层都在本地运行梯度计算和优化，因此每个层都有自己的优化器。作为标准选择，我们传递 `Adam` 优化器，并将默认学习率设置为 0.03，因为经过实验发现这是最佳速率。使用 `loss_var` 和 `loss_count` 变量跟踪损失。使用旧的 Layer Optimizer 来解决这个问题 https://github.com/keras-team/keras-io/issues/1241
    """


    def __init__(
        self,
        dims,
        layer_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.03),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_optimizer = layer_optimizer
        self.loss_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.loss_count = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.layer_list = [keras.Input(shape=(dims[0],))]
        for d in range(len(dims) - 1):
            self.layer_list += [
                FFDense(
                    dims[d + 1],
                    optimizer=self.layer_optimizer,
                    loss_metric=keras.metrics.Mean(),
                )
            ]

    # This function makes a dynamic change to the image wherein the labels are
    # put on top of the original image (for this example, as MNIST has 10
    # unique labels, we take the top-left corner's first 10 pixels). This
    # function returns the original data tensor with the first 10 pixels being
    # a pixel-based one-hot representation of the labels.
    """
    这个函数对图像进行动态修改，在原始图像的顶部放置标签（对于这个示例，由于 MNIST 具有 10 个唯一标签，我们取左上角的前 10 个像素）。该函数返回原始数据张量，其中前 10 个像素是标签的基于像素的 one-hot 表示。
    """

    @tf.function(reduce_retracing=True)
    def overlay_y_on_x(self, data):
        X_sample, y_sample = data
        max_sample = tf.reduce_max(X_sample, axis=0, keepdims=True)
        max_sample = tf.cast(max_sample, dtype=tf.float64)
        X_zeros = tf.zeros([10], dtype=tf.float64)
        X_update = xla.dynamic_update_slice(X_zeros, max_sample, [y_sample])
        X_sample = xla.dynamic_update_slice(X_sample, X_update, [0])
        return X_sample, y_sample

    # A custom `predict_one_sample` performs predictions by passing the images
    # through the network, measures the results produced by each layer (i.e.
    # how high/low the output values are with respect to the set threshold for
    # each label) and then simply finding the label with the highest values.
    # In such a case, the images are tested for their 'goodness' with all
    # labels.
    """
    自定义 `predict_one_sample` 函数通过将图像通过网络，测量每个层产生的结果（即每个标签相对于设定的阈值的输出值是多高/低），然后简单地找到具有最高值的标签来进行预测。在这种情况下，图像在所有标签上都被测试其“好度”。
    """

    @tf.function(reduce_retracing=True)
    def predict_one_sample(self, x):
        goodness_per_label = []
        x = tf.reshape(x, [tf.shape(x)[0] * tf.shape(x)[1]])
        for label in range(10):
            h, label = self.overlay_y_on_x(data=(x, label))
            h = tf.reshape(h, [-1, tf.shape(h)[0]])
            goodness = []
            for layer_idx in range(1, len(self.layer_list)):
                layer = self.layer_list[layer_idx]
                h = layer(h)
                goodness += [tf.math.reduce_mean(tf.math.pow(h, 2), 1)]
            goodness_per_label += [
                tf.expand_dims(tf.reduce_sum(goodness, keepdims=True), 1)
            ]
        goodness_per_label = tf.concat(goodness_per_label, 1)
        return tf.cast(tf.argmax(goodness_per_label, 1), tf.float64)

    def predict(self, data):
        x = data
        preds = list()
        preds = tf.map_fn(fn=self.predict_one_sample, elems=x)
        return np.asarray(preds, dtype=int)

    # This custom `train_step` function overrides the internal `train_step`
    # implementation. We take all the input image tensors, flatten them and
    # subsequently produce positive and negative samples on the images.
    # A positive sample is an image that has the right label encoded on it with
    # the `overlay_y_on_x` function. A negative sample is an image that has an
    # erroneous label present on it.
    # With the samples ready, we pass them through each `FFLayer` and perform
    # the Forward-Forward computation on it. The returned loss is the final
    # loss value over all the layers.
    """
    这个自定义的 `train_step` 函数覆盖了内部的 `train_step` 实现。我们将所有输入图像张量展平，然后在图像上产生正样本和负样本。正样本是带有正确标签编码的图像，使用 `overlay_y_on_x` 函数进行编码的。负样本是带有错误标签的图像。准备好样本后，我们将它们通过每个 `FFLayer`，并在其上执行 Forward-Forward 计算。返回的损失是所有层上的最终损失值。
    """

    @tf.function(jit_compile=True)
    def train_step(self, data):
        x, y = data

        # Flatten op
        x = tf.reshape(x, [-1, tf.shape(x)[1] * tf.shape(x)[2]])

        x_pos, y = tf.map_fn(fn=self.overlay_y_on_x, elems=(x, y))

        random_y = tf.random.shuffle(y)
        x_neg, y = tf.map_fn(fn=self.overlay_y_on_x, elems=(x, random_y))

        h_pos, h_neg = x_pos, x_neg

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, FFDense):
                print(f"Training layer {idx+1} now : ")
                h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg)
                self.loss_var.assign_add(loss)
                self.loss_count.assign_add(1.0)
            else:
                print(f"Passing layer {idx+1} now : ")
                x = layer(x)
        mean_res = tf.math.divide(self.loss_var, self.loss_count)
        return {"FinalLoss": mean_res}


"""
## Convert MNIST `NumPy` arrays to `tf.data.Dataset`

We now perform some preliminary processing on the `NumPy` arrays and then convert them
into the `tf.data.Dataset` format which allows for optimized loading.
## 将 MNIST `NumPy` 数组转换为 `tf.data.Dataset`

现在我们对 `NumPy` 数组进行一些预处理，然后将它们转换为 `tf.data.Dataset` 格式，这允许进行优化的加载。
"""

x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255
y_train = y_train.astype(int)
y_test = y_test.astype(int)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.batch(60000)
test_dataset = test_dataset.batch(10000)

"""
## Fit the network and visualize results

Having performed all previous set-up, we are now going to run `model.fit()` and run 250
model epochs, which will perform 50*250 epochs on each layer. We get to see the plotted loss
curve as each layer is trained.
## 拟合网络并可视化结果

完成所有前面的设置后，我们现在将运行 `model.fit()` 并运行 250 个模型周期，这将在每个层上执行 50x250 个周期。我们可以看到在每个层训练时绘制的损失曲线。
"""

model = FFNetwork(dims=[784, 500, 500])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.03),
    loss="mse",
    jit_compile=True,
    metrics=[keras.metrics.Mean()],
)

epochs = 250
history = model.fit(train_dataset, epochs=epochs)

"""
## Perform inference and testing

Having trained the model to a large extent, we now see how it performs on the
test set. We calculate the Accuracy Score to understand the results closely.
## 进行推断和测试

在很大程度上训练了模型后，我们现在看它在测试集上的表现。我们计算准确度分数以更好地了解结果。
"""

preds = model.predict(tf.convert_to_tensor(x_test))

preds = preds.reshape((preds.shape[0], preds.shape[1]))

results = accuracy_score(preds, y_test)

print(f"Test Accuracy score : {results*100}%")

plt.plot(range(len(history.history["FinalLoss"])), history.history["FinalLoss"])
plt.title("Loss over training")
plt.show()

"""
## Conclusion

This example has hereby demonstrated how the Forward-Forward algorithm works using
the TensorFlow and Keras packages. While the investigation results presented by Prof. Hinton
in their paper are currently still limited to smaller models and datasets like MNIST and
Fashion-MNIST, subsequent results on larger models like LLMs are expected in future
papers.

Through the paper, Prof. Hinton has reported results of 1.36% test accuracy error with a
2000-units, 4 hidden-layer, fully-connected network run over 60 epochs (while mentioning
that backpropagation takes only 20 epochs to achieve similar performance). Another run of
doubling the learning rate and training for 40 epochs yields a slightly worse error rate
of 1.46%

The current example does not yield state-of-the-art results. But with proper tuning of
the Learning Rate, model architecture (number of units in `Dense` layers, kernel
activations, initializations, regularization etc.), the results can be improved
to match the claims of the paper.
## 结论

本示例演示了如何使用 TensorFlow 和 Keras 包来实现前向传播算法。虽然 Hinton 教授在他们的论文中提出的调查结果目前仍局限于像 MNIST 和 Fashion-MNIST 这样的较小模型和数据集，但未来的论文预计将在更大的模型（如 LLMs）上得出结果。

通过该论文，Hinton 教授报告了使用 2000 个单元、4 个隐藏层、完全连接的网络在 60 个周期内运行，测试准确率误差为 1.36%（同时提到反向传播只需 20 个周期即可实现类似的性能）。另一次翻倍学习率并训练 40 个周期的运行结果误差率略高，为 1.46%。

当前的示例没有产生最先进的结果。但是通过适当调整学习率、模型结构（`Dense` 层中的单元数、内核激活、初始化、正则化等）、可以改进结果以匹配论文的要求。
"""