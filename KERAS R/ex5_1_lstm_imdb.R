library(keras)

max_features=20000 
maxlen=80
epochs=3 
batch_size=32

## Dataset
{
  dataset = dataset_imdb(num_words = max_features)
  
  c(x_train,y_train,x_test,y_test) %<-% list(dataset$train$x,dataset$train$y,
                                             dataset$test$x,dataset$test$y)
  x_train = pad_sequences(x_train,maxlen = maxlen)
  x_test = pad_sequences(x_test,maxlen = maxlen)
}

## build model
RNN_LSTM = function(maxlen,max_features) {
  x = layer_input(shape = c(NULL,maxlen)) 
  h = layer_embedding(object=x,input_dim = max_features,output_dim = 128) %>%
    layer_lstm(128,dropout = .2,recurrent_dropout = .2)
  y = layer_dense(object = h, units = 1,activation = 'sigmoid')
  
  model = keras_model(x,y)
  
  model %>%
    compile(loss = 'binary_crossentropy',
            optimizer = 'adam',
            metrics = 'accuracy')
}

## run model
RNN_LSTM(maxlen,max_features) %>%
  fit(x_train,y_train,
      epochs = epochs,batch_size = batch_size,
      validation_data = list(x_test,y_test))

#evaluate model 
RNN_LSTM(maxlen,max_features) %>%
  evaluate(x_test,y_test,
           batch_size = batch_size)
