rm(list=ls())

# install the keras & tensorflow packages
#----------------------------------------
# note: this only has to be done once

# see website https://keras.rstudio.com for info about keras
install.packages("keras")

library(keras)
install_keras()

# there is currently an issue with keras in R and
# the newest version of tensorflow
# so you'll need this fix
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

# visualize the digits
par(mfcol=c(6,6))
par(mar=c(0, 0.2, 3, 0.2), tck=0)

rand_samp <- sample.int(nrow(x_train), 36)

for (idx in rand_samp) { 
  im <- x_train[idx,,,1]
  im <- t(apply(im, 2, rev)) 
  image(1:28, 1:28, im, col=gray((0:255)/255), 
        xaxt='n', main=paste(y_train[idx]))
}

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
  layer_conv_2d(filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu',
                input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = 2) %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

# print model summary
summary(model)

# compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
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

# output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')

# predict the classes for the test data
y_pred <- model %>% predict_classes(x_test)

# reversing to_categorical for the confusion matrix
inverse_to_categorical <- function(mat){
  apply(mat, 1, function(row) which(row==max(row))-1)
}

y_test <- inverse_to_categorical(y_test)

# confusion matrix
table(y_test, y_pred)

# plot some of the wrongly classified digits
samp_wrong <- sample(which(y_test != y_pred), 36)

for (idx in samp_wrong) { 
  im <- x_test[idx,,,1]
  im <- t(apply(im, 2, rev)) 
  image(1:28, 1:28, im, col=gray((0:255)/255), 
        xaxt='n', main=paste("true=", y_test[idx], " pred=", y_pred[idx],sep=""))
}
