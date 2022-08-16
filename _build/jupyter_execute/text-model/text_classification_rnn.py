#!/usr/bin/env python
# coding: utf-8

# # 使用 RNN 做文本分类

# 文本分类任务使用 [recurrent neural network](https://developers.google.com/machine-learning/glossary/#recurrent_neural_network) 在 [IMDB large movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/) 数据机上训练情感分类任务。

# ## 安装依赖
# 
# 先来安装需要用到的依赖包。

# In[ ]:


get_ipython().system('pip install numpy tensorflow_datasets tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple')


# ## 导入依赖

# In[ ]:


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()


# 导入  `matplotlib` 并且创建一个帮主函数来画图:

# In[ ]:


import matplotlib.pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric], "")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_" + metric])


# ## 创建输入 Pipeline
# 
# IMDB 数据集是一个二分类的数据集，其中包含的评论信息有两种分类 *positive* 和 *negative* 的情感。
# 
# 使用 [TFDS](https://www.tensorflow.org/datasets) 下载数据集. 查看 [loading text tutorial](https://www.tensorflow.org/tutorials/load_data/text) 查看如何手动的家在数据。

# In[ ]:


dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset["train"], dataset["test"]

train_dataset.element_spec


# 这会返回 dataset 中的数据 (text, label pairs):

# In[ ]:


for example, label in train_dataset.take(1):
    print("text: ", example.numpy())
    print("label: ", label.numpy())


# 接下来，把训练集数据打乱并且创建批量的 `(text, label)` 对:

# In[ ]:


BUFFER_SIZE = 10000
BATCH_SIZE = 64


# In[ ]:


train_dataset = (
    train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# In[ ]:


for example, label in train_dataset.take(1):
    print("texts: ", example.numpy()[:3])
    print()
    print("labels: ", label.numpy()[:3])


# ## 创建文本编码器

# 在把数据输入到模型之前，我们需要先对数据进行预处理。最简单的去进行预处理的方法是使用 `TextVectorization` 层。
# 
# 创建层并且把数据集中的文本传入到这个层，使用 `.adapt` 函数进行计算:

# In[ ]:


VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))


# `.adapt` 函数设置 Tokenization 层的词典，这里我们打印前20个 token。

# In[ ]:


vocab = np.array(encoder.get_vocabulary())
vocab[:20]


# 一旦我们有了词典，接下来就可以使用它把 tokens 转换成 indices（字典索引）。字典索引的张量会进行 padding，使用 0 把每一个 batch 中的数据补齐成最长的长度（除非设置一个固定的 `output_sequence_length`）。

# In[ ]:


encoded_example = encoder(example)[:3].numpy()
encoded_example


# In[ ]:


for n in range(3):
    print("Original: ", example[n].numpy())
    print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
    print()


# ## 创建模型

# ![A drawing of the information flow in the model](../images/bidirectional.png)

# Above is a diagram of the model. 
# 
# 1. This model can be build as a `tf.keras.Sequential`.
# 
# 2. The first layer is the `encoder`, which converts the text to a sequence of token indices.
# 
# 3. After the encoder is an embedding layer. An embedding layer stores one vector per word. When called, it converts the sequences of word indices to sequences of vectors. These vectors are trainable. After training (on enough data), words with similar meanings often have similar vectors.
# 
#   This index-lookup is much more efficient than the equivalent operation of passing a one-hot encoded vector through a `tf.keras.layers.Dense` layer.
# 
# 4. A recurrent neural network (RNN) processes sequence input by iterating through the elements. RNNs pass the outputs from one timestep to their input on the next timestep.
# 
#   The `tf.keras.layers.Bidirectional` wrapper can also be used with an RNN layer. This propagates the input forward and backwards through the RNN layer and then concatenates the final output. 
# 
#   * The main advantage of a bidirectional RNN is that the signal from the beginning of the input doesn't need to be processed all the way through every timestep to affect the output.  
# 
#   * The main disadvantage of a bidirectional RNN is that you can't efficiently stream predictions as words are being added to the end.
# 
# 5. After the RNN has converted the sequence to a single vector the two `layers.Dense` do some final processing, and convert from this vector representation to a single logit as the classification output. 
# 

# The code to implement this is below:

# In[ ]:


model = tf.keras.Sequential(
    [
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True,
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)


# Please note that Keras sequential model is used here since all the layers in the model only have single input and produce single output. In case you want to use stateful RNN layer, you might want to build your model with Keras functional API or model subclassing so that you can retrieve and reuse the RNN layer states. Please check [Keras RNN guide](https://www.tensorflow.org/guide/keras/rnn#rnn_state_reuse) for more details.

# The embedding layer [uses masking](https://www.tensorflow.org/guide/keras/masking_and_padding) to handle the varying sequence-lengths. All the layers after the `Embedding` support masking:

# In[ ]:


print([layer.supports_masking for layer in model.layers])


# To confirm that this works as expected, evaluate a sentence twice. First, alone so there's no padding to mask:

# In[ ]:


# predict on a sample text without padding.

sample_text = (
    "The movie was cool. The animation and the graphics "
    "were out of this world. I would recommend this movie."
)
predictions = model.predict(np.array([sample_text]))
print(predictions[0])


# Now, evaluate it again in a batch with a longer sentence. The result should be identical:

# In[ ]:


# predict on a sample text with padding

padding = "the " * 2000
predictions = model.predict(np.array([sample_text, padding]))
print(predictions[0])


# Compile the Keras model to configure the training process:

# In[ ]:


model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"],
)


# ## Train the model

# In[ ]:


history = model.fit(
    train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30
)


# In[ ]:


test_loss, test_acc = model.evaluate(test_dataset)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


# In[ ]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, "accuracy")
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, "loss")
plt.ylim(0, None)


# Run a prediction on a new sentence:
# 
# If the prediction is >= 0.0, it is positive else it is negative.

# In[ ]:


sample_text = (
    "The movie was cool. The animation and the graphics "
    "were out of this world. I would recommend this movie."
)
predictions = model.predict(np.array([sample_text]))


# ## Stack two or more LSTM layers
# 
# Keras recurrent layers have two available modes that are controlled by the `return_sequences` constructor argument:
# 
# * If `False` it returns only the last output for each input sequence (a 2D tensor of shape (batch_size, output_features)). This is the default, used in the previous model.
# 
# * If `True` the full sequences of successive outputs for each timestep is returned (a 3D tensor of shape `(batch_size, timesteps, output_features)`).
# 
# Here is what the flow of information looks like with `return_sequences=True`:
# 
# ![layered_bidirectional](../images/layered_bidirectional.png)

# The interesting thing about using an `RNN` with `return_sequences=True` is that the output still has 3-axes, like the input, so it can be passed to another RNN layer, like this:

# In[ ]:


model = tf.keras.Sequential(
    [
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1),
    ]
)


# In[ ]:


model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"],
)


# In[ ]:


history = model.fit(
    train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30
)


# In[ ]:


test_loss, test_acc = model.evaluate(test_dataset)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


# In[ ]:


# predict on a sample text without padding.

sample_text = (
    "The movie was not good. The animation and the graphics "
    "were terrible. I would not recommend this movie."
)
predictions = model.predict(np.array([sample_text]))
print(predictions)


# In[ ]:


plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plot_graphs(history, "accuracy")
plt.subplot(1, 2, 2)
plot_graphs(history, "loss")


# Check out other existing recurrent layers such as [GRU layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU).
# 
# If you're interested in building custom RNNs, see the [Keras RNN Guide](https://www.tensorflow.org/guide/keras/rnn).
# 
