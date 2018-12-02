library(keras)

Nin = 784
Nh = 100
number_of_class = 10
Nout = number_of_class

##데이터 셋 
{
  dataset = dataset_mnist()
  c(x_train,y_train,x_test,y_test) %<-% list(dataset$train$x,dataset$train$y,
                                             dataset$test$x,dataset$test$y)
  
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  
  
  x_train = array_reshape(x_train,
                          dim = c(nrow(x_train),dim(x_train)[2]*dim(x_train)[3]))
  x_train = x_train/255
  x_test = array_reshape(x_test,
                         dim = c(nrow(x_test),dim(x_test)[2]*dim(x_test)[3]))
  x_test = x_test/255
}


## 모델 빌드1 ANN_model_sequential

ANN_seq_func = function() {
  model  = keras_model_sequential()
  model %>%
    layer_dense(units = Nh,input_shape = c(NULL,Nin),activation = 'relu') %>%
    layer_dense(units = Nout,activation = 'softmax') %>%
    compile(loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = 'accuracy')
}

## 모델 빌드2 ANN_model_class
ANN_model_class = function() {
  x = layer_input(shape = c(NULL,Nin))
  h = layer_dense(object = x,units = Nh) %>%
    layer_activation('relu')
  y = layer_dense(object = h,units = Nout) %>%
    layer_activation('softmax')
  
  model = keras_model(x,y)
  
  model %>%
    compile(loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = 'accuracy')
}

##모델 구동 및 평가
ANN_seq_func() %>%
  fit(x_train,y_train,
      batch_size = 100,
      epochs = 15) 

ANN_seq_func() %>%
  evaluate(x_test,y_test)

ANN_model_class() %>%
  fit(x_train,y_train,
      batch_size = 100,
      epochs = 15) %>%
  evaluate(x_test,y_test)
ANN_model_class() %>%
  evaluate(x_test,y_test)


###### 별도의 function 정의 없이 모델 빌드 

model  = keras_model_sequential()
model %>%
  layer_dense(units = Nh,input_shape = c(NULL,Nin),activation = 'relu') %>%
  layer_dense(units = Nout,activation = 'softmax') %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy') %>%
  fit(x_train,y_train,
      batch_size = 100,
      epochs = 15) 
model %>%
  evaluate(x_test,y_test)
