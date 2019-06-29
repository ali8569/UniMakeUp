package ir.markazandroid.unimakeup.facedetect;

import android.app.Activity;
import android.content.Context;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

import org.apache.commons.io.FileUtils;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import ir.markazandroid.unimakeup.R;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    public static final int NATIVE_DETECTOR = 1;

    private MenuItem mItemFace50;
    private MenuItem mItemFace40;
    private MenuItem mItemFace30;
    private MenuItem mItemFace20;
    private MenuItem mItemType;

    private Mat mRgba;
    private Mat mGray;
    private Mat mGrayScaled;
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;
    private DetectionBasedTracker mNativeDetector;

    private int mDetectorType = NATIVE_DETECTOR;
    private String[] mDetectorName;

    private float mRelativeFaceSize = 0.4f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("native-lib");
                    System.loadLibrary("c++_shared");

                    try {
                        // load cascade file from application resources
                        InputStream is = getAssets().open("lbpcascade_frontalface.xml");
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");

                        if (!mCascadeFile.exists()) {
                            FileOutputStream os = new FileOutputStream(mCascadeFile);

                            byte[] buffer = new byte[4096];
                            int bytesRead;
                            while ((bytesRead = is.read(buffer)) != -1) {
                                os.write(buffer, 0, bytesRead);
                            }
                            is.close();
                            os.close();
                        }

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    FaceDetector faceDetector;

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

         mOpenCvCameraView = (JavaCameraView) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        File path = new File(Environment.getExternalStorageDirectory(), "modelData.dat");
        if (!path.exists()) {
            try {
                FileUtils.copyInputStreamToFile(getAssets().open("shape_predictor_68_face_landmarks.dat"), path);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        faceDetector = new FaceDetector();
        Log.e("FaceDetect", faceDetector.isFaceLandmarksDetectorReady() + "");
        faceDetector.prepareFaceLandmarksDetector(path.getPath());
        faceDetector.prepareFaceDetector();
        Log.e("FaceDetect", faceDetector.isFaceLandmarksDetectorReady() + "");
        mOpenCvCameraView.setCameraIndex(0);
        mOpenCvCameraView.enableFpsMeter();

       /* try {
            CameraManager manager  = (CameraManager) getSystemService(CAMERA_SERVICE);
            CameraCharacteristics cameraCharacteristics = null;
            cameraCharacteristics = manager.getCameraCharacteristics("0");
            StreamConfigurationMap streamConfigurationMap = cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            android.util.Size[] sizes = streamConfigurationMap.getOutputSizes(ImageReader.class);

            for (android.util.Size size : sizes) {
                Log.e("prev Size", size.getWidth() + "--" + size.getHeight());
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }*/

       /* Camera camera = Camera.open(0);
        List<Camera.Size> previewSizes = camera.getParameters().getSupportedPreviewSizes();
        for (Camera.Size size : previewSizes) {
            Log.e("prev Size", size.width + "--" + size.height);
        }
        previewSizes = camera.getParameters().getSupportedPictureSizes();
        for (Camera.Size size : previewSizes) {
            Log.e("img Size", size.width + "--" + size.height);
        }
        previewSizes = camera.getParameters().getSupportedVideoSizes();
        if (previewSizes != null)
            for (Camera.Size size : previewSizes) {
                Log.e("vid Size", size.width + "--" + size.height);
            }

        List<Integer> formats = camera.getParameters().getSupportedPreviewFormats();
        for (Integer format : formats) {

            Log.e("format", format + "");
        }

        List<int[]> fpsRange = camera.getParameters().getSupportedPreviewFpsRange();
        for (int[] fpsR : fpsRange) {
            Log.e("fps range", fpsR[0] + "--" + fpsR[1]);
        }*/

    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
        mGrayScaled = new Mat();
        points = new MatOfPoint2f();
        faces = new MatOfRect();

        points.alloc(136);
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        mGrayScaled.release();
        points.release();
        faces.release();
    }

    private int resize = 3;
    MatOfPoint2f points;
    MatOfRect faces;

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        Log.e("start of frame time", "");

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        Imgproc.resize(mGray, mGrayScaled, new Size(), 1.0 / resize, 1.0 / resize);

        //Core.transpose(mRgba, mRgba);
        //Core.flip(mRgba, mRgba, -1);

        //Core.transpose(mGray, mGray);
        //Core.flip(mGray, mGray, -1);


        if (mAbsoluteFaceSize == 0) {
            int height = mGrayScaled.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }


// Resize image for face detection


        long now = System.currentTimeMillis();
        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGrayScaled, faces, 1.2, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        } else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGrayScaled, faces);
        } else {
            Log.e(TAG, "Detection method is not selected!");
        }
        long duration = System.currentTimeMillis() - now;
        Rect[] facesArray = faces.toArray();
        if (facesArray.length > 0) {
            Log.d("Detect time", duration + "");
            facesArray[0].x *= resize;
            facesArray[0].y *= resize;
            facesArray[0].width *= resize;
            facesArray[0].height *= resize;
            Imgproc.rectangle(mRgba, facesArray[0], FACE_RECT_COLOR, 3);

            long now2 = System.currentTimeMillis();
            faceDetector.detectPoints(mGrayScaled.nativeObj, points.nativeObj,
                    Math.round(facesArray[0].tl().x / resize),
                    Math.round(facesArray[0].tl().y / resize),
                    Math.round(facesArray[0].br().x / resize),
                    Math.round(facesArray[0].br().y / resize));
            duration = System.currentTimeMillis() - now2;

            Point[] pointsArray = points.toArray();
            Log.d("point java time", duration + "");

            for (int i = 0; i < 136; i += 2) {
                Point point = new Point(pointsArray[i].x * resize, pointsArray[i + 1].x * resize);
                Imgproc.circle(mRgba, point, 2, FACE_RECT_COLOR, 3);
            }

            //points.release();
        }

        duration = System.currentTimeMillis() - now;
        Log.d("whole time", duration + "");
        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }
}
