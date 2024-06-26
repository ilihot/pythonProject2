import tensorflow as tf
import matplotlib.pyplot as plt

# Data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Model parameter
W = tf.Variable(tf.random.normal([1]), name='weight')


# Hypothesis function
def hypothesis(x):
    return x * W


# Cost function
def cost_function(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


# Optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Initialize lists to store W values and corresponding cost values
W_val = []
cost_val = []

# Evaluate cost for a range of W values
for i in range(-30, 50):
    feed_W = i * 0.1
    W.assign([feed_W])  # Assign new value to W
    curr_cost = cost_function(hypothesis(x_train), y_train).numpy()
    W_val.append(W.numpy()[0])
    cost_val.append(curr_cost)

# Plot the cost function
plt.plot(W_val, cost_val)
plt.xlabel('W value')
plt.ylabel('Cost')
plt.title('Cost function for different W values')
plt.show()