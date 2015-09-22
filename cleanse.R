require(dplyr)
require(caret)
require(rpart)
require(rattle)

cleanData <- function(df) {
  cd <- select(df, classe,
           roll_belt,pitch_belt, yaw_belt, total_accel_belt,
           gyros_belt_x, gyros_belt_y, gyros_belt_z,
           accel_belt_x, accel_belt_y, accel_belt_z,
           magnet_belt_x, magnet_belt_y, magnet_belt_z,
           roll_arm, pitch_arm, yaw_arm, total_accel_arm, 
           gyros_arm_x, gyros_arm_y, gyros_arm_z,
           accel_arm_x, accel_arm_y, accel_arm_z,
           magnet_arm_x, magnet_arm_y, magnet_arm_z,
           roll_dumbbell, pitch_dumbbell, yaw_dumbbell)
  cd
}

pml <- read.csv("data/pml-training.csv")
pmlt <- read.csv("data/pml-testing.csv")

ctraining <- cleanData(pml)
ctesting <- cleanData(pmlt)