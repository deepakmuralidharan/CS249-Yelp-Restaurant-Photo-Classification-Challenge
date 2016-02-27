# CS249-Big-Data-Analytics
Project for CS249

# Steps to Run the file

1. First execute “run_serialize.py”. This will generate the CNN codes for the training and testing images and the features will be stored as X_train, y_train, X_test, y_test in the workspace.

2. Then execute “transfer_yelp_restaurant_softmax.py”. This will load the features X_train, X_test, and labels y_train/ y_test and run the classification algorithm.

 
# Some other stuff

3. Generating features: Generate features for a subset of images in training and testing restaurant dataset and store in X_train, X_test. (This consumes most of the time —- around 12 hours on CPU —- and is independent of the label).

2. Training the classifier and fine-tuning for a specific attribute: We need to assign labels to the images i.e we need to classify each image as to whether it belongs to “restaurant_is_expensive” (label 1) or doesn’t belong to it (label 1). 

3. We need to perform Step (2) for all nine attributes individually and calculate the F1 score, precision, recall etc.

4. The main thing we need to look at is the “do_train” function in transfer_yelp_restaurant_softmax.py which is the training of the last layer. Here for the standard example, cross-entropy loss is used. However, if we want to generate a nine labels at a single go, we need to change the loss entropy and also the labels for each image. The labels for each image will now be a one hot encoded vector.


