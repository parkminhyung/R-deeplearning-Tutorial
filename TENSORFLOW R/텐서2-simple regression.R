library(tensorflow)

#데이터설정
x<-as.numeric(c(1:10))
y<-2*x+3+rnorm(10, mean = 0, sd = .5)

#플롯
plot(x,y,xlim = c(0,10),ylim = c(0,30))

#변수설정
W<-tf$Variable(tf$constant(1.0),name = "Weight")
b<-tf$Variable(tf$constant(2.0),name = 'bias')

#모델설정
y_hat<-x*W+b

#cost function
cost<-tf$reduce_mean(tf$square(y-y_hat))
train<-optimizer$minimize(cost)

sess<-tf$Session()
sess$run(tf$global_variables_initializer())

#train
for (i in 1:5000) {
  sess$run(train)
  if (i %% 10 == 0 ) print( c(sess$run(W), sess$run(b)) )
}

#checking result
# Check our answer
sess$run(W)
sess$run(b)
abline(sess$run(b), sess$run(W)) #plot에 선추가

# Close session
sess$close()
