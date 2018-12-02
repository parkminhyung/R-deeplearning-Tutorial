library(keras)

###setting parameters
{
  num_classes = 10
  batch_size = 128
  epochs = 10
}

#데이터 준비
{
  datasets = dataset_mnist()
  c(x_train,y_train,x_test,y_test) %<-% list(datasets$train$x,datasets$train$y,
                                             datasets$test$x,datasets$test$y)
  
  y_train = to_categorical(y_train,num_classes)
  y_test = to_categorical(y_test,num_classes)
  
  c(img_rows,img_cols) %<-% c(dim(x_train)[2:3])
  
  if(k_image_data_format() == "channels_first"){
    x_train = array(data = x_train,dim = c(nrow(x_train),1,img_rows,img_cols)) 
    x_test = array(data = x_test, dim = c(nrow(x_test),1,img_rows,img_cols)) 
    input_shape = c(1,img_rows,img_cols)
  } else {
    x_train = array(data = x_train,dim = c(nrow(x_train),img_rows,img_cols,1)) 
    x_test = array(data = x_test, dim = c(nrow(x_test),img_rows,img_cols,1)) 
    input_shape = c(img_rows,img_cols,1)
  }
}
  
##모델빌드 ####
{
  model = keras_model_sequential()
  
  model %>%
    layer_conv_2d(32,kernel_size = c(3,3),activation = 'relu',
                  input_shape = input_shape,) %>%
    layer_conv_2d(64,kernel_size = c(3,3),activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(rate = .25) %>%
    layer_flatten() %>%
    layer_dense(units = 128,activation = 'relu') %>%
    layer_dropout(rate = .5) %>%
    layer_dense(num_classes,activation = 'softmax') %>%
    compile(loss = 'categorical_crossentropy',
            optimizer = 'rmsprop')
}

## 모델구동 및 평가 ##
{
  model %>%
    fit(x_train,y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_split = .2)
  
  model %>% 
    evaluate(x_test,y_test)
}


