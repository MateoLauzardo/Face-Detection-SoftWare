package com.example;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import java.io.File;
import java.util.ArrayList;
import static org.bytedeco.opencv.global.opencv_core.*;



public class FaceRecognizer {

    public static void main(String[] args) {

        //takes a look at folder named training, and gets list of all files inside folder
        File folder = new File("training");
        File[] imageFiles = folder.listFiles();

        //checks if folder is empty
        if (imageFiles == null || imageFiles.length == 0) {
            System.out.println("No training images found!");
            return;
        }

        //array list of Mats (images related to OpenCV)
        ArrayList<Mat> images = new ArrayList<>();
        //creates matrix to store the labels for each image
        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        IntPointer labelsPtr = new IntPointer(labels.data());


        int counter = 0;
        for (File file : imageFiles) {
            //for each file in folder
            if (file.getName().endsWith(".jpg") || file.getName().endsWith(".png")) {
                //Loads the image from the file in grayscale (black and white)
                Mat img = opencv_imgcodecs.imread(file.getAbsolutePath(), opencv_imgcodecs.IMREAD_GRAYSCALE);

                //if image failed to load
                if (img.empty()) {
                    System.out.println("Failed to load image: " + file.getName());
                    continue;
                }

                images.add(img); //stores successful loaded image to the images list
                labelsPtr.put(counter, 1);  //assigns a label 1 to image
                counter++;
            }
        }


        //used to store multiple images needed for training
        MatVector imagesVector = new MatVector(images.size());
        for (int i = 0; i < images.size(); i++) {
            imagesVector.put(i, images.get(i)); //fills "imagesVector" with all images from list
        }


        LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create();
        recognizer.train(imagesVector, labels); //trains the face using images and their labels
        recognizer.save("trained_model.xml"); //saves trained face to a file to be loaded later

        System.out.println("Training complete. Model saved as 'trained_model.xml'");
    }
}