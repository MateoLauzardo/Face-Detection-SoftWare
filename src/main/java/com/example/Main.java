package com.example;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;


public class Main {
    public static void main(String[] args) {


        //Loads a Haar cascade XML file used for face detection.
        CascadeClassifier faceDetector = new CascadeClassifier("resources/haarcascade_frontalface_alt.xml");


        //loads and reads recognizer using Local Binary Patterns Histograms (LBPH)
        LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create();
        recognizer.read("trained_model.xml");

        //opens webcam
        VideoCapture capture = new VideoCapture(1);
        if (!capture.isOpened()) {
            System.err.println("Cannot open webcam");
            return;
        }

        //holds each capture color image
        Mat frame = new Mat();
        //gold the grayscale version for face detection
        Mat gray = new Mat();


        //loops endlessly until user press quit
        while (true) {
            if (!capture.read(frame) || frame.empty()) {
                System.out.println("No frame captured");
                break;
            }

            cvtColor(frame, gray, COLOR_BGR2GRAY); //converts color image to grayscale

            //detect faces in grayscale images
            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(gray, faces);

            //loops over each detected face, and grabs region of interest ROI
            for (int i = 0; i < faces.size(); i++) {
                Rect faceRect = faces.get(i);
                Mat faceROI = new Mat(gray, faceRect);


                //Resize faces to 100x100
                Mat resizedFace = new Mat();
                resize(faceROI, resizedFace, new Size(100, 100));


                //Uses recognizer to predict the face's label and confidence level.
                IntPointer label = new IntPointer(1);
                DoublePointer confidence = new DoublePointer(1);
                recognizer.predict(faceROI, label, confidence);


                //Extracts label and confidence values from pointers / to use individually
                int predictedLabel = label.get(0);
                double conf = confidence.get(0);


                Scalar color;
                if (predictedLabel == 1 && conf < 70) {
                    color = new Scalar(0, 255, 0, 0);  // green
                } else {
                    color = new Scalar(0, 0, 255, 0);  // red
                }

                //draws rectangle on face
                rectangle(frame, faceRect, color, 2, LINE_8, 0);

                //writes label above face
                String text = (predictedLabel == 1 && conf < 70) ? "You" : "Unknown";
                putText(frame, text, new Point(faceRect.x(), faceRect.y() - 10),
                        FONT_HERSHEY_SIMPLEX, 1.0, color);
            }


            imshow("Face Recognition", frame);


            int key = waitKey(30);
            if (key == 'q' || key == 27) { // q or ESC to quit
                break;
            }


        }//end of while loop

        capture.release();
        destroyAllWindows();
    }
}