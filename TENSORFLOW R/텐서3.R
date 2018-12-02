library(tensorflow)

x<-tf$placeholder(tf$float32,shape(10L,1L))
y<-tf$placeholder(tf$float32,shape(10L,1L))

#W&b
W<-tf$Variable(tf$constant(1.0,shape = c(1L,1L)))
b<-tf$Variable(tf$constant(2.0,shape = c(1L,1L)))

hypothesis = tf$matmul(x,W)+b

#cost function
cost <- tf$reduce_mean(tf$square(y-hypothesis))

#optimize
optimizer<-tf$train$GradientDescentOptimizer(.01)
train<-optimizer$minimize(cost)

#session 초기화
sess <- tf$Session()
sess$run(tf$global_variables_initializer())

#train
x.data<-matrix(c(1:10),ncol=1)
y.data<-2 * x.data + 3 + rnorm(10, mean = 0, sd = 0.5)
result <- matrix(0, nrow = 2000, ncol = 3)

for (i in 1:3000) {
  result[i, ] <- unlist(sess$run(c(cost, W, b, train),
                                 feed_dict = dict(x = x.data, y = y.data)) )
  if (i %% 10 == 0 ) print(result[i, 2:3])
}

# Check our answer
# Cost function decreasing
plot(c(1000:2000), result[c(1000:2000),1])

# Let's see what our data looks like?
plot(x.data, y.data, xlim = c(0, 10), ylim = c(0, 30))

sess$run(W)
sess$run(b)
abline(sess$run(b), sess$run(W))

# Close session
sess$close()
