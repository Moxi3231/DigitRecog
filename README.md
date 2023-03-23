# Handwritten Digit Recognition


Github Link: https://github.com/Moxi3231/DigitRecog
## Introduction
The aim of this project is to identify handwritten digits from both web camera (live source) and saved video file. The project follows the following steps to detect handwritten digits:

1. Get a frame from the video
2. Convert the frame into a grayscale image
3. Apply Canny edge filter on the grayscale image
4. Get contours from the edged image
5. Finally, get the bounding box for contours (Region of Interest)
6. To preserve aspect ratio, convert rectangular bounding boxes into a square bounding box by simply selecting the maximum dimension of width and length
7. Once a square bounding box is procured, resize it to 28*28 to make it compatible with the CNN model
    - 7.1 Additional preprocessing steps are taken here to get better result
    - 7.2 Before resizing the digit (bouding box), the digit was centered in the square bounding box, and then dilation operation was applied to match the output from canny edge with dataset on which CNN model is trained.
8. After resizing the digit, pass it to the model, and write prediction on the digit along with bounding box


#
- > Note: To run code first time or to re-train the model, in DigitIdentifer.py file, get_trained_model function's parameter must be true
    - > get_trained_model(training = True) at line 7
#
## Model Architecture
For this project, a custom CNN model with architecture proposed in the second lecture by Dr. Harry Li (CMPE 258) was used. The model summary and accuracy of the model at the end on the test set are given below:

    Model:
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)             (32, 26, 26, 32)          320       
                                                                                
    max_pooling2d (MaxPooling2D  (32, 13, 13, 32)         0         
    )                                                               
                                                                                
    conv2d_1 (Conv2D)           (32, 11, 11, 16)          4624      
                                                                                
    max_pooling2d_1 (MaxPooling  (32, 5, 5, 16)           0         
    2D)                                                             
                                                                                
    conv2d_2 (Conv2D)           (32, 3, 3, 8)             1160      
                                                                                
    flatten (Flatten)           (32, 72)                  0         
                                                                                
    dense (Dense)               (32, 32)                  2336      
                                                                                
    dense_1 (Dense)             (32, 10)                  330       
                                                                                
    =================================================================
    Total params: 8,770
    Trainable params: 8,770
    Non-trainable params: 0
    _________________________________________________________________
    Model Accuracy of Test Set (In Percentage): 98.73

### Output Screenshot
Below is the output screenshot of the model:

![Alt text](./Screenshot%202023-03-22%20at%2011.15.38%20PM.png "Handwritten Digit Recognition Output")

### Conclusion
- > This project successfully identifies handwritten digits from both web camera and saved video file. 
- > The custom CNN model with the proposed architecture achieved an accuracy of 98.73% on the test set.