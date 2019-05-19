rm(list=ls())
# in this exercise you will need to add your own code where indicated

# set working directory
#----------------------
# 'setwd' and 'img_dir' (directory where the image data is stored) appropriately for your computer, e.g.
setwd("<path on your computer>/StatPred_2019")
img_dir <- "<path on your computer>/StatPred_2019/bbbc021v1_subset/"

library(keras)

# data preparation
batch_size <- 32
epochs <- 30
num_classes <- 6

# input image dimensions
input_shape <- c(256, 256, 3)
target_size <- c(256, 256) # for generators

# generator for loading images
data_generator <- image_data_generator(rescale = 1/255, validation_split = 0.25)

train_generator <- flow_images_from_directory(img_dir, generator = data_generator,
                                              target_size = target_size, color_mode = "rgb",
                                              class_mode = "categorical", batch_size = batch_size,
                                              subset = "training")

valid_generator <- flow_images_from_directory(img_dir, generator = data_generator,
                                              target_size = target_size, color_mode = "rgb",
                                              class_mode = "categorical", batch_size = batch_size,
                                              subset = "validation")

# define model
## ADD YOUR OWN CODE HERE TO DEFINE THE MODEL

summary(model)
## you should have 1,121,062 parameters in the model


# compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# train model
model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(train_generator$n/batch_size),
  validation_data = valid_generator, 
  validation_steps = as.integer(valid_generator$n/batch_size),
  epochs = epochs)
