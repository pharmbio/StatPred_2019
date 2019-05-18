rm(list=ls())

# install the keras & tensorflow packages
#----------------------------------------
# note: this only has to be done once

# see website https://keras.rstudio.com for info about keras
install.packages("keras")

library(keras)
install_keras()

# if you're using a mac you'll need this fix
library(tensorflow)
install_tensorflow(version = "1.12")


# MNIST example
#--------------
# just run this example as is, you need not add or modify any of the code
# example based heavily upon https://keras.rstudio.com/articles/examples/mnist_cnn.html
library(keras)

# data preparation
batch_size <- 64
num_classes <- 10
epochs <- 10

# input image dimensions
img_rows <- 28
img_cols <- 28

# the data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# redefine dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# define model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

# print model summary
summary(model)

# compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# train model
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

scores <- model %>% evaluate(
  x_test, y_test, verbose = 0
)

# Output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')
