library(keras)

x = as.matrix(c(0,1,2,3,4))
y = 2*x+1

##model build 
model = keras_model_sequential()
model %>%
  layer_dense(input_shape = c(NULL,1),
              units = 1) %>%
  compile(loss = 'mse',
          optimizer = 'sgd')

model %>%
  fit(x[1:2],y[1:2],
      epochs = 1000,
      verbose = 0)

model %>%
  predict(x[3:5])
