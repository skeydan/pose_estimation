library(R.matlab)

# The file joints.mat is a MATLAB data file containing the joint annotations in a 3x14x2000 matrix
# called 'joints' with x and y locations and a binary value indicating the visbility of each joint.
# The ordering of the joints is as follows: 
  # Right ankle
  # Right knee
  # Right hip
  # Left hip
  # Left knee
  # Left ankle
  # Right wrist
  # Right elbow
  # Right shoulder
  # Left shoulder
  # Left elbow
  # Left wrist
  # Neck
 # Head top

## y axis starts from the top

annotations <- readMat("joints.mat")
joints <- annotations$joints
dim(joints)

joints[ , , 1]






# crop
# (i) We crop the images in all datasets to a resolution of 256 × 256.
# For training images in all datasets, we
# crop around the person’s center computed by using the ground-truth joint positions. For test
# images in all datasets we crop around the rough person location when available, otherwise
# we crop around the center of the image. 


# data augmentation

# We use scaling # ∈ {0.5, 1.5}, translation ∈ {−20, 20}, rotation ∈ {−20 ◦ , 20 ◦ } and horizontal flipping with
# probability 0.5 for data augmentation.
# Each form of augmentation, i.e, scaling, rotation, translation, reflection is an image warp operation. 
# The warps are represented by a warp matrix containing random values of scaling, rotation, flip and
# translation. The matrix is then used to compute the pixel values in warped image through
# a combination of backward warping and an interpolation method. The warped joints are
# computed by multiplication of warp matrix with unwarped joints.
# The naive application of the above procedure results in applying the warp around the
# upper left image corner and the person of interest may be located elsewhere in the image.
# Our goal is to apply the warp around the center position of the person of interest. To this
# end, we apply a warp W that is computed as follows:

# The warp on the right first shifts the perturbed center of person (x p + t x , y p + t y ) to the upper
# left corner of the image. The perturbed operation implicitly produces the effect of applying
# a random translation. The middle warp then applies random scaling s, rotation(cosθ , sinθ )
# and reflection r . The left warp then shifts back the warped person to the center (c x , c y ) of
# the cropped image. Figure 3(b) illustrates the procedure.
# One caveat with extensive augmentation is that in many random warps some warped
# joints will be outside of the image boundary. One solution is to use less augmentation. How-
#   ever, this can have a significant impact on performance. To this end, we keep on randomly
# generating warps and then only select the warps for which all warped joints end up inside
# the image boundary.




