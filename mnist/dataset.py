from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnistdata/', one_hot = True)

print("training data size: ", mnist.train._num_examples)
print("validating data size: ", mnist.validation._num_examples)
print("testing data size: ", mnist.test._num_examples)
print("example data: ", mnist.train.images[0])
print("example data label: ", mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print("x shape:", xs.shape)
print("y shape:", ys.shape)