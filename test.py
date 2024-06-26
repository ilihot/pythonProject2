# hyerim

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# import tensorflow as tf
# import matplotlib.pyplot as plt


# print(tf.__version__)

#
# "hello" 텐서 생성
# hello = tf.constant("hello, tensorflow")
#
# 텐서 출력
# print(hello.numpy().decode('utf-8'))


#print(sess.run(hello))
#
# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0, tf.float32)
# node3 = tf.add(node1, node2)
#
# print(node1.numpy(), node2.numpy())
# print(node3.numpy())


# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a + b


#@tf.function 데코레이터: TensorFlow 연산을 그래프 모드로 변환하여 성능을 최적화합니다.
#
# def add_tensors(a, b):
#     return a+b
#
# #create tensors
# a = tf.constant(3.0, dtype = tf.float32)
# b = tf.constant(4.0, dtype = tf.float32)
#
# #add the tensors and print the result
# result = add_tensors(a, b)
# print(result.numpy())
#
# #add another pari of tensors
# a = tf.constant([1.0, 3.0], dtype = tf.float32)
# b = tf.constant([2.0, 4.0], dtype = tf.float32)
#
# result = add_tensors(a, b)
# print(result.numpy())

#0606 study ML lab 02 - TensorFlow로 간단한 linear regression을 구현 (new)

#첫번째 예제
#
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]
#
# W = tf.Variable(tf.random.normal([1]), name= 'weight') #variable = trainable
# b = tf.Variable(tf.random.normal([1]), name= 'bias')
#
# #hypothesis
# def hypothesis(x):
#     return x * W + b # the predicted output. It multiplies the input x by the weight (W) and adds the bias (b)
#
# def cost_function(y_pred, y_true):
#     return tf.reduce_mean(tf.square(y_pred - y_true))

# #traning step
# def train_step(x,y): #train_step 함수는 x와 y를 입력으로 받고, tf.GradientTape()를 사용하여 그래디언트를 계산하는 동안 코드 블록을 실행합니다
#     with tf.GradientTape() as tape: #as tape는 tf.GradientTape() 객체를 tape라는 변수에 할당합니다., "Gradient tape context"는 TensorFlow에서 모델의 그래디언트를 계산하는 도구로, 주로 모델 학습 시 사용
#         pred = hypothesis(x)
#         cost = cost_function(pred, y) #the predicted values (pred) and the true values (y).
#     gradients = tape.gradient(cost, [W, b]) #The tape.gradient method computes the gradients of the cost with respect to the model parameters (W and b). This is where automatic differentiation occurs.
#     optimizer.apply_gradients(zip(gradients, [W, b])) #optimizer.apply_gradients method updates the model parameters (W and b) using the computed gradients. This step is part of the optimization process.
#     return cost
#
# # initializes the optimizer. Here, it's using Stochastic Gradient Descent (SGD) as the optimizer with a learning rate of 0.01.
# optimizer = tf.optimizers.SGD(learning_rate=0.01)
#
# #Traning
# for step in range (2001):
#     cost_val = train_step (x_train, y_train)
#     if step % 20 == 0:
#         print(step, cost_val.numpy(), W.numpy(), b.numpy())
#
#

#=======================================================================================
#예제2
# W = tf.Variable(tf.random.normal([1]), name= 'weight') #variable = trainable
# b = tf.Variable(tf.random.normal([1]), name= 'bias')
# # X = tf.placeholder(tf.float32, shape = [None]) TensorFlow 2.x or above, where tf.placeholder is not available
# # Y = tf.placeholder(tf.float32, shape = [None])
# #  typically use tf.data.Dataset or directly pass the data to your model without the need for placeholders.
# x_train = [1, 2, 3, 4, 5]
# y_train = [2.1, 3.1, 4.1, 5.1, 6.1]
#
# def hypothesis(x):
#     return x * W + b # the predicted output. It multiplies the input x by the weight (W) and adds the bias (b)
#
# def cost_function(y_pred, y_true):
#     return tf.reduce_mean(tf.square(y_pred - y_true))
# # tf.square(y_pred - y_true) calculates the squared difference between the predicted values (y_pred) and the true values (y_true).
# # tf.reduce_mean calculates the mean of all the squared differences, yielding the mean squared error (MSE), which is commonly used as the cost function in regression problems.
#
# def train_step(x,y): #train_step 함수는 x와 y를 입력으로 받고, tf.GradientTape()를 사용하여 그래디언트를 계산하는 동안 코드 블록을 실행합니다
#     with tf.GradientTape() as tape: #as tape는 tf.GradientTape() 객체를 tape라는 변수에 할당합니다., "Gradient tape context"는 TensorFlow에서 모델의 그래디언트를 계산하는 도구로, 주로 모델 학습 시 사용
#         pred = hypothesis(x)
#         cost = cost_function(pred, y) #the predicted values (pred) and the true values (y).
#     gradients = tape.gradient(cost, [W, b]) #The tape.gradient method computes the gradients of the cost with respect to the model parameters (W and b). This is where automatic differentiation occurs.
#     optimizer.apply_gradients(zip(gradients, [W, b])) #optimizer.apply_gradients method updates the model parameters (W and b) using the computed gradients. This step is part of the optimization process.
#     return cost
#
# # initializes the optimizer. Here, it's using Stochastic Gradient Descent (SGD) as the optimizer with a learning rate of 0.01.
# optimizer = tf.optimizers.SGD(learning_rate=0.01)
#
# #Traning
# for step in range(2001):
#     cost_val = train_step(x_train, y_train)
#     # feed_dict = {X: [1, 2, 3, 4, 5],
#     #              Y: [2.1, 3.1, 4.1, 5.1, 6.1]}
#     if step % 20 == 0:
#         print(step, cost_val.numpy(), W.numpy(), b.numpy())
#


#=======================================================================================
#예제3 ML lab 03 - Linear Regression 의 cost 최소화의 TensorFlow 구현 (new)
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
