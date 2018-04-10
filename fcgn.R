library(keras)

# following: Rafi et al., An Efficient Convolutional Network for Human Pose Estimation

# adaptation of the batch normalized GoogleNet architecture

## We use the first 17 layers and
## remove the average pooling, drop-out, linear and soft-max layers from the last stages of the network
## We add a skip connection to combine feature maps from layer 13 with feature maps from layer 17.
## We upsample the feature maps from layer 17 to the resolution of the feature maps from layer 13 by a deconvolution filter of both size and stride 2.
## The output of FCGN consists of coarse feature maps from layer 13 and 17 that have 16 times lesser resolution than the input image due to max/average pooling by a factor of 16.


model_weights_exist <- FALSE
weights_file <- ""

batch_size <- 10
num_epochs <- 1
learning_rate <- 0.001

# Inception module ------------------------------------------------------------------------------------------


# inception module
inception_module <- function(prev_layer,
                            path1,
                            path2,
                            path3,
                            path4)
  
{
  if (!is.null(path1)) {
    conv1 <- prev_layer %>%
      layer_conv_2d(
        kernel_size = path1$kernel_size,
        filters = path1$filters,
        padding = "same",
        activation = "relu"
      )
  }
  
  conv2 <-
    prev_layer %>% layer_conv_2d(
      kernel_size = path2$c1$kernel_size,
      filters = path2$c1$filters,
      padding = "same",
      activation = "relu"
    ) %>%
    layer_conv_2d(
      kernel_size = path2$c2$kernel_size,
      filters = path2$c2$filters,
      padding = "same",
      activation = "relu"
    )
  conv3 <-
    prev_layer %>% layer_conv_2d(
      kernel_size = path3$c1$kernel_size,
      filters = path3$c1$filters,
      padding = "same",
      activation = "relu"
    ) %>%
    layer_conv_2d(
      kernel_size = path3$c2$kernel_size,
      filters = path3$c2$filters,
      padding = "same",
      activation = "relu"
    )           %>%
    layer_conv_2d(
      kernel_size = path3$c3$kernel_size,
      filters = path3$c3$filters,
      padding = "same",
      activation = "relu"
    )
  
  conv4 <- prev_layer %>%
    (
      if (path4$p$pooling == "avg")
        layer_average_pooling_2d(
          pool_size = path4$p$poolsize,
          strides = c(1, 1),
          padding = "same"
        )
      else
        layer_max_pooling_2d(
          pool_size = path4$p$poolsize,
          strides = c(1, 1),
          padding = "same"
        )
    )
  if (!is.null(path4$c)) {
    conv4 <- conv4 %>%
      layer_conv_2d(
        kernel_size = path4$c$kernel_size,
        filters = path4$c$filters,
        padding = "same",
        activation = "relu"
      )
  }
    
    # default axis for concatenate is -1, so this concatenates along the depth dimension
    output <- layer_concatenate(if (!is.null(l <- get0("conv1")))
      list(l, conv2, conv3, conv4)
      else
        list(conv2, conv3, conv4))
}



# input layer -------------------------------------------------------------


## We crop the images in all datasets to a resolution of 256 × 256. For training images in all datasets, we
# crop around the person’s center computed by using the ground-truth joint positions. For test
# images in all datasets we crop around the rough person location when available, otherwise
# we crop around the center of the image.

input_tensor <- layer_input(shape = c(256, 256, 3))



# Model -------------------------------------------------------------------


output_tensor <- input_tensor %>%
  
  # 1
  # 128*128
  layer_conv_2d(
    filters = 64,
    kernel_size = c(7, 7),
    padding = "same",
    strides = c(2, 2)
  ) %>%
  layer_batch_normalization() %>%
  layer_activation_elu() %>%
  
  # 2
  # 64*64*64
  layer_max_pooling_2d(pool_size = c(3, 3),
                       strides = c(2, 2),
                       padding = "same") %>%
  
  # 3
  # 64*64*64
  layer_conv_2d(filters = 64,
                kernel_size = c(1, 1),
                padding = "same") %>%
  layer_batch_normalization() %>%
  layer_activation_elu() %>%
  
  # 4
  # 64*64*192
  layer_conv_2d(filters = 192,
                kernel_size = c(3, 3),
                padding = "same") %>%
  layer_batch_normalization() %>%
  layer_activation_elu() %>%
  
  # 5
  # 32*32*192
  layer_max_pooling_2d(pool_size = c(3, 3),
                       strides = c(2, 2),
                       padding = "same")

output_tensor

# 6
# 32*32*256
output_tensor <- output_tensor %>%
  inception_module(
    path1 = list(kernel_size = 1, filters = 64),
    path2 = list(
      c1 = list(kernel_size = 1, filters = 64),
      c2 = list(kernel_size = 3, filters = 64)
    ),
    path3 = list(
      c1 = list(kernel_size = 1, filters = 64),
      c2 = list(kernel_size = 3, filters = 96),
      c3 = list(kernel_size = 3, filters = 96)
    ),
    path4 = list(
      p = list(pooling = "avg", poolsize = 3),
      c = list(kernel_size = 1, filters = 32)
    )
  ) %>%
  
  # 7
  # 32*32*256
  inception_module(
    path1 = list(kernel_size = 1, filters = 64),
    path2 = list(
      c1 = list(kernel_size = 1, filters = 64),
      c2 = list(kernel_size = 3, filters = 64)
    ),
    path3 = list(
      c1 = list(kernel_size = 1, filters = 96),
      c2 = list(kernel_size = 3, filters = 96),
      c3 = list(kernel_size = 3, filters = 96)
    ),
    path4 = list(
      p = list(pooling = "avg", poolsize = 3),
      c = list(kernel_size = 1, filters = 32)
    )
  ) %>%
  
  # 8
  # 32*32*512
  inception_module(
    path1 = NULL,
    path2 = list(
      c1 = list(kernel_size = 1, filters = 128),
      c2 = list(kernel_size = 3, filters = 160)
    ),
    path3 = list(
      c1 = list(kernel_size = 1, filters = 64),
      c2 = list(kernel_size = 3, filters = 96),
      c3 = list(kernel_size = 3, filters = 96)
    ),
    path4 = list(p = list(pooling = "max", poolsize = 3),
                 c = NULL)
  ) %>%

  # 9
  # 16*16*512
  layer_max_pooling_2d(pool_size = c(3, 3), strides = c(2, 2), padding = "same") %>%

  # 10
  # 16*16*576
  inception_module(
    path1 = list(kernel_size = 1, filters = 224),
    path2 = list(
      c1 = list(kernel_size = 1, filters = 64),
      c2 = list(kernel_size = 3, filters = 96)
    ),
    path3 = list(
      c1 = list(kernel_size = 1, filters = 96),
      c2 = list(kernel_size = 3, filters = 128),
      c3 = list(kernel_size = 3, filters = 128)
    ),
    path4 = list(
      p = list(pooling = "avg", poolsize = 3),
      c = list(kernel_size = 1, filters = 128)
    )
  ) %>%

  # 11
  # 16*16*576
  inception_module(
    path1 = list(kernel_size = 1, filters = 192),
    path2 = list(
      c1 = list(kernel_size = 1, filters = 96),
      c2 = list(kernel_size = 3, filters = 128)
    ),
    path3 = list(
      c1 = list(kernel_size = 1, filters = 96),
      c2 = list(kernel_size = 3, filters = 128),
      c3 = list(kernel_size = 3, filters = 128)
    ),
    path4 = list(
      p = list(pooling = "avg", poolsize = 3),
      c = list(kernel_size = 1, filters = 128)
    )
  ) %>%

  # 12
  #16*16*576
  inception_module(
    path1 = list(kernel_size = 1, filters = 160),
    path2 = list(
      c1 = list(kernel_size = 1, filters = 128),
      c2 = list(kernel_size = 3, filters = 160)
    ),
    path3 = list(
      c1 = list(kernel_size = 1, filters = 128),
      c2 = list(kernel_size = 3, filters = 160),
      c3 = list(kernel_size = 3, filters = 160)
    ),
    path4 = list(
      p = list(pooling = "avg", poolsize = 3),
      c = list(kernel_size = 1, filters = 96)
    )
  ) %>%

  # 13
  # 16*16*576
  inception_module(
    path1 = list(kernel_size = 1, filters = 96),
    path2 = list(
      c1 = list(kernel_size = 1, filters = 128),
      c2 = list(kernel_size = 3, filters = 192)
    ),
    path3 = list(
      c1 = list(kernel_size = 1, filters = 160),
      c2 = list(kernel_size = 3, filters = 192),
      c3 = list(kernel_size = 3, filters = 192)
    ),
    path4 = list(
      p = list(pooling = "avg", poolsize = 3),
      c = list(kernel_size = 1, filters = 96)
    )
  )

# skip connection
output_skip <- output_tensor

output <- output_tensor %>%

  # 14
  # 16*16*576
  inception_module(
    path1 = NULL,
    path2 = list(
      c1 = list(kernel_size = 1, filters = 128),
      c2 = list(kernel_size = 3, filters = 192)
    ),
    path3 = list(
      c1 = list(kernel_size = 1, filters = 192),
      c2 = list(kernel_size = 3, filters = 256),
      c3 = list(kernel_size = 3, filters = 256)
    ),
    path4 = list(p = list(pooling = "max", poolsize = 3),
                 c = NULL)
  ) %>%
  
  # 15
  # 16*16*576
  layer_max_pooling_2d(pool_size = c(3, 3), strides = c(2, 2)) %>%

  # 16
  # 16*16*576
  inception_module(
    path1 =  list(kernel_size = 1, filters = 352),
    path2 = list(
      c1 = list(kernel_size = 1, filters = 192),
      c2 = list(kernel_size = 3, filters = 320)
    ),
    path3 = list(
      c1 = list(kernel_size = 1, filters = 160),
      c2 = list(kernel_size = 3, filters = 224),
      c3 = list(kernel_size = 3, filters = 224)
    ),
    path4 = list(
      p = list(pooling = "avg", poolsize = 3),
      c = list(kernel_size = 1, filters = 128)
    )
  ) %>%

  # 17
  # 16*16*576
  inception_module(
    path1 = list(kernel_size = 1, filters = 352),
    path2 = list(
      c1 = list(kernel_size = 1, filters = 192),
      c2 = list(kernel_size = 3, filters = 320)
    ),
    path3 = list(
      c1 = list(kernel_size = 1, filters = 192),
      c2 = list(kernel_size = 3, filters = 224),
      c3 = list(kernel_size = 3, filters = 224)
    ),
    path4 = list(
      p = list(pooling = "max", poolsize = 3),
      c = list(kernel_size = 1, filters = 128)
    )
  ) %>%

  # 18
  # 16*16*576
  layer_conv_2d(filters = 192, kernel_size = c(3, 3)) %>%
  layer_batch_normalization() %>%
  layer_activation_elu()


output_tensor
 
 # don't add as in resnet, concatenate on depth dimension!
# output_tensor <- layer_add(list(output_skip, output_tensor))
# 16*16*1152
output_tensor <- layer_concatenate(list(output_skip, output_tensor))


model <- keras_model(input_tensor, output_tensor)
model %>% summary()

model %>% compile(optimizer = optimizer_adam(lr = learning_rate),
                loss = loss_binary_crossentropy)




# Data --------------------------------------------------------------------




# Model training ----------------------------------------------------------

## We train the network from scratch without any pre-training with a learning rate of 0.001 with an exponential decay of 0.96 applied
# every 50 epochs. We train the network for 120 epochs for each dataset.





