library(tensorflow)

# Given data
X.data <- cbind(1:6, c(4,2,1,5,7,9)) #6*2 matrix
X.data <- cbind(1, X.data) # 6*3 matrix
X.data 

y.data <- matrix(c(0, 0, 0, 1, 1, 1), ncol = 1) #6*1 matrix
y.data

#placeholder setting
X <- tf$placeholder(tf$float32, shape(6L, 3L))
y <- tf$placeholder(tf$float32, shape(6L, 1L))

# W setting as parameter
W <- tf$Variable(tf$constant(c(1.0,1.0,1.0),shape = c(3L,1L)),name = 'parameter')

#calculating hypothesis and applying sigmod function
Hypothesis<-tf$sigmoid(tf$matmul(X,W))

#cost function
cost<- -tf$reduce_mean(y*tf$log(Hypothesis)+(1-y)*tf$log(1-Hypothesis))

#optimize
optimizer<-tf$train$GradientDescentOptimizer(learning_rate = .1)
train<-optimizer$minimize(cost)

#initialize session
sess<-tf$Session()
sess$run(tf$global_variables_initializer())

#training
result<-matrix(0,nrow = 5000,ncol = 4)
for (i in 1:5000) {
  result[i,] <- unlist(sess$run(c(cost,W,train),
                                feed_dict = dict(X=X.data,y=y.data)))
}

result<-cbind(1:5000,result)
colnames(result)<-c("step","cost","b","W1","W2")
result
head(result)
tail(result)

#plotting
plot(1:5000,result[1:5000,2])
W.est<-as.vector(result[5000,3:5])

#calculating result
result_sig<-round(sess$run(tf$sigmoid(X.data %*% W.est)))
result_sig
y.data

accuracy<-cor(result_sig,y.data)
accuracy


sess$close()
