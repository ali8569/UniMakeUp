package ir.markazandroid.unimakeup;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.FrameLayout;

import com.badlogic.gdx.backends.android.AndroidApplication;
import com.badlogic.gdx.backends.android.AndroidApplicationConfiguration;

import org.apache.commons.io.FileUtils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.Executors;

import ir.markazandroid.unimakeup.facedetect.FaceDetector;

public class AndroidLauncher extends Activity {

	static {System.loadLibrary("opencv_java4");}
	@Override
	protected void onCreate (Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(new FrameLayout(this));
		//AndroidApplicationConfiguration config = new AndroidApplicationConfiguration();

		File path = new File(Environment.getExternalStorageDirectory(),"modelData.dat");
		try {
			if (!path.exists())
				FileUtils.copyInputStreamToFile(getAssets().open("shape_predictor_68_face_landmarks.dat"),path);
			FaceDetector faceDetector = new FaceDetector();
			Log.e("FaceDetect",faceDetector.isFaceLandmarksDetectorReady()+"");
			faceDetector.prepareFaceLandmarksDetector(path.getPath());
			faceDetector.prepareFaceDetector();
			Log.e("FaceDetect",faceDetector.isFaceLandmarksDetectorReady()+"");

			path = new File(Environment.getExternalStorageDirectory(),"demo_pic.jpg");
			if (!path.exists())
				FileUtils.copyInputStreamToFile(getAssets().open("demo_pic.jpg"),path);

			Mat mat = Imgcodecs.imread(path.getPath(),Imgcodecs.IMREAD_GRAYSCALE);
			//Imgproc.cvtColor(mat,mat,Imgproc.COLOR_RGB2BGR);
			MatOfPoint2f points = new MatOfPoint2f();
			points.alloc(68);

			long start = System.currentTimeMillis();
			faceDetector.detectLandmark(mat.getNativeObjAddr(),points.getNativeObjAddr());
			long end = System.currentTimeMillis();

			Log.e("Duration",(end-start)+"");


			List<Point> points1 = points.toList();

			Executors.newCachedThreadPool().execute(() -> {
				long start1 = System.currentTimeMillis();
				faceDetector.detectLandmark(mat.getNativeObjAddr(),points.getNativeObjAddr());
				long end1 = System.currentTimeMillis();

				Log.e("Duration",(end1 - start1)+"");
			});
			//new Thread().start();

			File mCascadeFile = new File(Environment.getExternalStorageDirectory(), "lbpcascade_frontalface.xml");
			if(!mCascadeFile.exists()){
				FileUtils.copyInputStreamToFile(getAssets().open("lbpcascade_frontalface.xml"),mCascadeFile);
			}

			CascadeClassifier mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());

			MatOfRect faces = new MatOfRect();

			float mRelativeFaceSize   = 0.5f;
			int   mAbsoluteFaceSize   = 0;

			if (Math.round(250 * mRelativeFaceSize) > 0) {
				mAbsoluteFaceSize = Math.round(250 * mRelativeFaceSize);
			}

			Imgproc.resize(mat,mat,new Size(),0.4,0.4);

			Log.e("mat Size",mat.width()+"");

			long start1 = System.currentTimeMillis();
			mJavaDetector.detectMultiScale(mat, faces, 1.4, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
					new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
			long end1 = System.currentTimeMillis();
			if (!faces.empty()) Log.e("cv","detected");
			Log.e("Duration cv",(end1 - start1)+"");
			points1=null;
		} catch (IOException e) {
			e.printStackTrace();
		}


		//initialize(new LiveView(), config);

		//Gdx.gl20.glGetString(GL20.GL_EXTENSIONS);

	}



}
