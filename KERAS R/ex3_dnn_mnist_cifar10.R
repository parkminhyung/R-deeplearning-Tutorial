library(keras)

#parameter
{
  Pd_l=c(0.0, 0.0)
  Nh_l = c(100, 50)
  number_of_class = 10
  Nout = number_of_class
}

#dataset
{
  dataset = dataset_cifar10()
  
  c(x_train,y_train,x_test,y_test) %<-% list(dataset$train$x, dataset$train$y,
                                             dataset$test$x, dataset$test$y)
  
  c(W,H,C) %<-% c(dim(x_train)[2:4])
  x_train = array_reshape(x_train,dim = c(nrow(x_train),W*H*C))
  x_train = x_train/255
  x_test = array_reshape(x_test,dim = c(nrow(x_test),W*H*C))
  x_test = x_test/255
  
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  
  Nin=W*H*C
}

#build model 
DNN = function(Nin,Pd_l,Nh_l,Nout) {
  model = keras_model_sequential()
  model %>%
    layer_dense(units = Nh_l[1],activation = 'relu',input_shape = c(NULL,Nin),name = 'Hidden-1') %>%
    layer_dropout(rate = Pd_l[1]) %>%
    layer_dense(units = Nh_l[2],activation = 'relu',name = 'Hidden-2') %>%
    layer_dropout(rate = Pd_l[2]) %>%
    layer_dense(units = Nout, activation = 'softmax') %>%
    compile(loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = 'accuracy')
}

#run model 
DNN(Nin,Pd_l,Nh_l,Nout) %>%
  fit(x_train,y_train,
      epochs = 100,batch_size = 100,
      validation_split = .2)

#evaluate model
DNN(Nin,Pd_l,Nh_l,Nout) %>%
  evaluate(x_test,y_test,batch_size=100)


