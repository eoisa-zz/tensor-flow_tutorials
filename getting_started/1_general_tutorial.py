import tensorflow as tf

node_1 = tf.constant(3.0, dtype=tf.float32)
node_2 = tf.constant(4.0)

sess = tf.Session()

print(sess.run([node_1, node_2]))

node_3 = tf.add(node_1, node_2)

print(sess.run(node_3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_node = a + b    #  the '+' is an overload of tf.add(a, b)

print(sess.run(add_node, {a: 3, b: 4.5}))
print(sess.run(add_node, {a: [1, 3], b: [2, 4]}))

tripple_add_node = add_node * 3

print(sess.run(tripple_add_node, {a: 3, b: 4.5}))

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W*x + b

init = tf.global_variables_initializer()

sess.run(init)  #   Starts a fresh run

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fix_W = tf.assign(W, [-1])  #   Wow, look! Magical nubmers that get our loss to zero, if only
fix_b = tf.assign(b, [1])   #   there was a way to make an algorithm that was smart
                            #   enough to figure this out on its own...

sess.run([fix_W, fix_b])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)  #   Resets the values to their defaults

#   Run the training algorithm x1000
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b])) #   Take a look at the final values of W and b



