
Useful Terms-
CNN Codes-> output of the penultimate layer(size 1x1x2048), that will be given as the input to the last classification layer.
one-hot encoded vector->If the labels assigned to an image are 2,5,7 then the one-hot encoded vector for that image is written as [0,0,1,0,0,1,0,1,0].

Methodology-
a) Generate CNN codes from the training images by randomly sampling a batch of 250 images.Along with this, one hot encoded vector for the image labels is also generated and stored.These CNN codes and one-hot encoded vectors are stored as .npy files.This process is repeated for all the images with a sample size of 250.Finally we get 925 .npy files for CNN codes and 925 .npy files for the one-hot encoded labels.

b) These 925 .npy files were then used to generate 'conc_CNN_code.npy' which contains the output after concatenation of CNN codes from 925 .npy files and other file 'conc_CNN_label.npy' which contains the output after concatenation of one-hot encoded vectors from 925 .npy files.

c) The concatenated .npy files in last step were used in the main.py to train the last classification layer and generate the predicted labels of the testset for each of the nine labels and find the cross validation accuracy,training accuarcy and cross-entropy loss values.

d) The predicted labels and ground truth were stored separately and were used to obtain the F1-score,precision ,recall values and other graphs.


Steps to follow-
1.Run 'new_image_run_serialize.py' for step (a) of methodology.Set the correct path of the trainig images on line number 78.This will produce 925 .npy files for one hot encoded labels in the same folder like y_train1.npy,y_train2.npy,...,y_traink.npy and 925 .npy files for CNN codes like X_train1.npy,X_train2.npy,...,X_traink.npy.Since this takes a lot of time to run like 26 hours , we have given the concatenated file i.e. the output of step (b) of the methodology.

2.Run 'main.py' 9 times for each of the label.The label number has to be changed at line numbers 31 and 33 of main.py for running it 9 times for different labels.This will generate the plots for cross-entropy values,training accuracy and validation accuracy.It will also store 'predicted_class.npy' and 'truth_class.npy'.Since running this file 9 times for different labels takes a lot of time, we have provided with the predicted_class.npy and truth_class.npy for the all the nine labels in the folder 'files_for_f1_calc'.

3.Run 'test.py' for generating the F1-scores,precision,recall and other graphs.


  
