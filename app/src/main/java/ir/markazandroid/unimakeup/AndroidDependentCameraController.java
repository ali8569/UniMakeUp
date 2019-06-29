package ir.markazandroid.unimakeup;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.media.Image;
import android.media.ImageReader;
import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import android.view.SurfaceView;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.Mesh;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.VertexAttribute;
import com.badlogic.gdx.graphics.VertexAttributes;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShaderProgram;
import com.quickbirdstudios.yuv2mat.Yuv;

import org.apache.commons.io.FileUtils;
import org.opencv.android.JavaCamera2View;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import ir.markazandroid.unimakeup.facedetect.DetectionBasedTracker;
import ir.markazandroid.unimakeup.facedetect.FaceDetector;

/**
 * Coded by Ali on 5/26/2019.
 */
@SuppressWarnings("deprecation")
public class AndroidDependentCameraController implements Camera.PreviewCallback {

    private static byte[] image; //The image buffer that will hold the camera image when preview callback arrives
    private static byte[] yuvImageY;
    private static byte[] yuvImageU;
    private static byte[] yuvImageV;
    private static byte[] rgbaBytes;

    private Camera camera; //The camera object

    private int width = 1280;
    private int height = 720;

    private Mat frame, rgbaFrame, rgbaFrameRotated;

    //The Y and UV buffers that will pass our image channel data to the textures
    private ByteBuffer yBuffer;
    private ByteBuffer uvBuffer;
    private ByteBuffer rgbaBuffer;
    private Context context;

    ShaderProgram shader; //Our shader
    Texture yTexture; //Our Y texture
    Texture uvTexture; //Our UV texture
    Texture rgbaTexture;
    Mesh mesh; //Our mesh that we will draw the texture on

    static {
        System.loadLibrary("opencv_java4");
    }

    public AndroidDependentCameraController(Context context) {
        this.context = context;

        //Our YUV image is 12 bits per pixel
        image = new byte[width * height / 8 * 12];
        yuvImageY = new byte[width * height];
        yuvImageU = new byte[width * height/2];
        yuvImageV = new byte[width * height/2];
        rgbaBytes = new byte[width * height * 4];
        frame = new Mat(height + (height / 2), width, CvType.CV_8UC1);
        rgbaFrame = new Mat(height, width, CvType.CV_8UC4);
        rgbaFrameRotated = new Mat(width, height, CvType.CV_8UC4);
    }

    public void init() {

        /*
         * Initialize the OpenGL/libgdx stuff
         */

        //Allocate textures
        yTexture = new Texture(width, height, Pixmap.Format.Intensity); //A 8-bit per pixel format
        uvTexture = new Texture(width / 2, height / 2, Pixmap.Format.LuminanceAlpha); //A 16-bit per pixel format
        rgbaTexture = new Texture(width, height, Pixmap.Format.RGBA8888); //A 8-bit per pixel format


        //Allocate buffers on the native memory space, not inside the JVM heap
        yBuffer = ByteBuffer.allocateDirect(width * height);
        uvBuffer = ByteBuffer.allocateDirect(width * height / 2); //We have (width/2*height/2) pixels, each pixel is 2 bytes
        rgbaBuffer = ByteBuffer.allocateDirect(width * height * 4);
        yBuffer.order(ByteOrder.nativeOrder());
        uvBuffer.order(ByteOrder.nativeOrder());
        rgbaBuffer.order(ByteOrder.nativeOrder());

        //Our vertex shader code; nothing special
        String vertexShader =
                "attribute vec4 a_position;                         \n" +
                        "attribute vec2 a_texCoord;                         \n" +
                        "varying vec2 v_texCoord;                           \n" +

                        "void main(){                                       \n" +
                        "   gl_Position = a_position;                       \n" +
                        "   v_texCoord = a_texCoord;                        \n" +
                        "}                                                  \n";

        //Our fragment shader code; takes Y,U,V values for each pixel and calculates R,G,B colors,
        //Effectively making YUV to RGB conversion
        String fragmentShader =
                "#ifdef GL_ES                                       \n" +
                        "precision highp float;                             \n" +
                        "#endif                                             \n" +

                        "varying vec2 v_texCoord;                           \n" +
                        // "uniform sampler2D y_texture;                       \n" +
                        // "uniform sampler2D uv_texture;                      \n" +
                        "uniform sampler2D rgba_texture;                      \n" +

                        "void main (void){                                  \n" +
                        //"   float r, g, b, y, u, v;                         \n" +
                        "   float r, g, b, a;                         \n" +

                        //We had put the Y values of each pixel to the R,G,B components by GL_LUMINANCE,
                        //that's why we're pulling it from the R component, we could also use G or B
                        //"   y = texture2D(y_texture, v_texCoord).r;         \n" +

                        //We had put the U and V values of each pixel to the A and R,G,B components of the
                        //texture respectively using GL_LUMINANCE_ALPHA. Since U,V bytes are interspread
                        //in the texture, this is probably the fastest way to use them in the shader
                        // "   u = texture2D(uv_texture, v_texCoord).a - 0.5;  \n" +
                        // "   v = texture2D(uv_texture, v_texCoord).r - 0.5;  \n" +


                        //The numbers are just YUV to RGB conversion constants
                        "   r = texture2D(rgba_texture, v_texCoord).r;         \n" +
                        "   g = texture2D(rgba_texture, v_texCoord).g;         \n" +
                        "   b = texture2D(rgba_texture, v_texCoord).b;         \n" +
                        "   a = texture2D(rgba_texture, v_texCoord).a;         \n" +
                        //"   r = y + 1.13983*v;                              \n" +
                        //"   g = y - 0.39465*u - 0.58060*v;                  \n" +
                        //"   b = y + 2.03211*u;                              \n" +

                        //We finally set the RGB color of our pixel
                        "   gl_FragColor = vec4(r, g, b, a);              \n" +
                        "}                                                  \n";

        //Create and compile our shader
        shader = new ShaderProgram(vertexShader, fragmentShader);

        //Create our mesh that we will draw on, it has 4 vertices corresponding to the 4 corners of the screen
        mesh = new Mesh(true, 4, 6,
                new VertexAttribute(VertexAttributes.Usage.Position, 2, "a_position"),
                new VertexAttribute(VertexAttributes.Usage.TextureCoordinates, 2, "a_texCoord"));

        //The vertices include the screen coordinates (between -1.0 and 1.0) and texture coordinates (between 0.0 and 1.0)
        float[] vertices = {
                -1.0f, 1.0f,   // Position 0
                0.0f, 0.0f,   // TexCoord 0
                -1.0f, -1.0f,  // Position 1
                0.0f, 1.0f,   // TexCoord 1
                1.0f, -1.0f,  // Position 2
                1.0f, 1.0f,   // TexCoord 2
                1.0f, 1.0f,   // Position 3
                1.0f, 0.0f    // TexCoord 3
        };

        //The indices come in trios of vertex indices that describe the triangles of our mesh
        short[] indices = {0, 1, 2, 0, 2, 3};

        //Set vertices and indices to our mesh
        mesh.setVertices(vertices);
        mesh.setIndices(indices);



        /*
         * Initialize the Android camera
         */
        setupCamera2();

        initDetector();

        Handler handler = new Handler(context.getMainLooper());

        handler.post(() -> camera2Api.openCamera(width,height));

        calculateParams(Gdx.graphics.getWidth(),Gdx.graphics.getHeight());

    }

    Camera2Api camera2Api;

    private void setupCamera2(){
        camera2Api=new Camera2Api(context, reader -> {

            Image image = reader.acquireLatestImage();
            if (image==null) return;
            if (isComputing) {
                image.close();
                return;
            }
            isComputing=true;

            Image.Plane[] planes = image.getPlanes();

            final int yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            Image.Plane Y = image.getPlanes()[0];
            Image.Plane U = image.getPlanes()[1];
            Image.Plane V = image.getPlanes()[2];

            int Yb = Y.getBuffer().remaining();
            int Ub = U.getBuffer().remaining();
            int Vb = V.getBuffer().remaining();

            Y.getBuffer().get(AndroidDependentCameraController.yuvImageY, 0, Yb);
            U.getBuffer().get(AndroidDependentCameraController.yuvImageU, Yb, Ub);
            V.getBuffer().get(AndroidDependentCameraController.yuvImageV, Yb+ Ub, Vb);

            FaceDetector.convertYUV420ToARGB8888(
                    yuvImageY,
                    yuvImageU,
                    yuvImageV,
                    rgbaBytes,
                    width,
                    height,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    false);


            image.close();
            Gdx.graphics.requestRendering();
        });
    }

    byte[] yArray=new byte[width*height];

    Mat y_mat = new Mat(height, width, CvType.CV_8UC1);
    Mat uv_mat = new Mat(height / 2, width / 2,CvType.CV_8UC2);

    private void setupCamera1(){
        camera = Camera.open(0);

        //We set the buffer ourselves that will be used to hold the preview image
        camera.setPreviewCallbackWithBuffer(this);

        //Set the camera parameters
        Camera.Parameters params = camera.getParameters();
        //params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
        params.setPreviewSize(width, height);
        params.set("orientation", "landscape");
        camera.setParameters(params);
        camera.setDisplayOrientation(90);


        Handler handler = new Handler(context.getMainLooper());
        handler.post(() -> {
            try {
                camera.setPreviewTexture(new SurfaceTexture(100));
            } catch (IOException e) {
                e.printStackTrace();
            }
        });


        //Set the first buffer, the preview doesn't start unless we set the buffers
        camera.addCallbackBuffer(image);


        //Start the preview
        camera.startPreview();
    }

    private Mat mGrayScaled;
    private File mCascadeFile;
    private DetectionBasedTracker mNativeDetector;
    private FaceDetector faceDetector;

    private int resize = 3;
    private MatOfPoint2f points;
    private MatOfRect faces;
    private Mat mGray;

    private void initDetector() {
        System.loadLibrary("native-lib");
        System.loadLibrary("c++_shared");
        try {
            // load cascade file from application resources
            InputStream is = context.getAssets().open("haarcascade_frontalface_default.xml");
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");

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

            mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);
            mNativeDetector.start();
            cascadeDir.delete();

            File path = new File(Environment.getExternalStorageDirectory(), "modelData.dat");
            if (!path.exists()) {
                try {
                    FileUtils.copyInputStreamToFile(context.getAssets().open("shape_predictor_68_face_landmarks.dat"), path);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            faceDetector = new FaceDetector();
            Log.e("FaceDetect", faceDetector.isFaceLandmarksDetectorReady() + "");
            faceDetector.prepareFaceLandmarksDetector(path.getPath());
            faceDetector.prepareFaceDetector();
            Log.e("FaceDetect", faceDetector.isFaceLandmarksDetectorReady() + "");

            points = new MatOfPoint2f();
            faces = new MatOfRect();

            points.alloc(136);

            mGrayScaled = new Mat();
            mGray=new Mat();



        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private float fromX, fromY, vWidth, vHeight,displayWidth,displayHeight;
    private float scaleFactor;
    private int cropSide; //0 no crop; 1 cropped width; 2 cropped height


    private void calculateParams(float displayWidth, float displayHeight) {

        //TODO Rotation
        /*int temp = width;
        width=height;
        height=temp;*/
        //
        this.displayWidth=displayWidth;
        this.displayHeight=displayHeight;
        float scaleWidth = displayWidth / width;
        float scaleHeight = displayHeight / height;

        if (scaleWidth > scaleHeight)
            cropSide = 2;
        else if (scaleWidth < scaleHeight)
            cropSide = 1;
        else
            cropSide = 0;

        scaleFactor = Math.max(scaleHeight, scaleWidth);
        vWidth = width* scaleFactor;
        vHeight = height* scaleFactor;

        float diff = (vWidth - displayWidth) / scaleFactor;
        fromX = diff / 2;

        diff = (vHeight - displayHeight) / scaleFactor;
        fromY = diff / 2;


/*

        switch (cropSide) {
            case 0:
                break; //fits the window

            case 1:

                break;

            case 2:

        }
*/

        Log.e("calculatedSizes", String.format(
                "frameWidth: %d -- frameHeight: %d\n" +
                        "displayWidth: %.2f -- displayHeight: %.2f\n" +
                        "scaleFactor: %.2f\n" +
                        "vWidth: %.2f -- vHeight: %.2f\n" +
                        "cropSize: %d\n" +
                        "fromX: %.2f -- fromY: %.2f"
                , width, height
                , displayWidth, displayHeight
                , scaleFactor
                , vWidth, vHeight
                , cropSide
                , fromX, fromY));
    }

    @Override
    public void onPreviewFrame(byte[] data, Camera camera) {


        //Send the buffer reference to the next preview so that a new buffer is not allocated and we use the same space
        Gdx.graphics.requestRendering();
        camera.addCallbackBuffer(image);


        //Log.e("Here","here");
    }

    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);

    private volatile boolean isComputing;

    public void drawRgba(SpriteBatch batch) {
        isComputing=true;
        //frame.put(0, 0, image);
        //Imgproc.cvtColor(frame, rgbaFrame, Imgproc.COLOR_YUV2RGBA_NV21, 4);

        //Core.transpose(rgbaFrame, rgbaFrameRotated);
        //Core.flip(rgbaFrameRotated, rgbaFrameRotated, -1);
        //int temp = width;
        //width=height;
        //height=temp;
        //calculateParams(displayWidth,displayHeight);

        //Core.rotate(rgbaFrame,rgbaFrameRotated,Core.ROTATE_90_COUNTERCLOCKWISE);

        //face detect
        Imgproc.cvtColor(rgbaFrame, mGray, Imgproc.COLOR_RGBA2GRAY, 1);
        Imgproc.resize(mGray, mGrayScaled, new Size(), 1.0 / resize, 1.0 / resize);

        int factor = Math.max(mGrayScaled.rows(),mGrayScaled.cols());

        mNativeDetector.setMinFaceSize(Math.round(factor * 0.4f));

        long now = System.currentTimeMillis();

        mNativeDetector.detect(mGrayScaled, faces);

        long duration = System.currentTimeMillis() - now;
        Rect[] facesArray = faces.toArray();
        if (facesArray.length > 0) {
            Log.d("Detect time", duration + "");
            facesArray[0].x *= resize;
            facesArray[0].y *= resize;
            facesArray[0].width *= resize;
            facesArray[0].height *= resize;
            Imgproc.rectangle(rgbaFrame, facesArray[0], FACE_RECT_COLOR, 2);

            long now2 = System.currentTimeMillis();
            faceDetector.detectPoints(mGray.nativeObj, points.nativeObj,
                    Math.round(facesArray[0].tl().x),
                    Math.round(facesArray[0].tl().y),
                    Math.round(facesArray[0].br().x),
                    Math.round(facesArray[0].br().y));
            duration = System.currentTimeMillis() - now2;

            Point[] pointsArray = points.toArray();
            Log.d("point java time", duration + "");

            for (int i = 0; i < 136; i += 2) {
                Point point = new Point(pointsArray[i].x, pointsArray[i + 1].x);
                Imgproc.circle(rgbaFrame, point, 1, FACE_RECT_COLOR, 1);
            }

            //points.release();
        }

        duration = System.currentTimeMillis() - now;
        Log.d("whole time", duration + "");

        rgbaFrame.get(0, 0, rgbaBytes);

        rgbaBuffer.put(rgbaBytes, 0, width * height * 4);
        rgbaBuffer.position(0);

        Gdx.gl.glActiveTexture(GL20.GL_TEXTURE0);
        rgbaTexture.bind();

        //TODO Rotation
        Gdx.gl.glTexImage2D(GL20.GL_TEXTURE_2D, 0, GL20.GL_RGBA, width, height, 0, GL20.GL_RGBA, GL20.GL_UNSIGNED_BYTE, rgbaBuffer);

        //Use linear interpolation when magnifying/minifying the texture to areas larger/smaller than the texture size
        //Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_MIN_FILTER, GL20.GL_LINEAR);
        //Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_MAG_FILTER, GL20.GL_LINEAR);
        //Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_WRAP_S, GL20.GL_CLAMP_TO_EDGE);
        //Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_WRAP_T, GL20.GL_CLAMP_TO_EDGE);

        //shader.begin();

        //Set the uniform y_texture object to the texture at slot 0
        //shader.setUniformi("rgba_texture", 0);

        //Render our mesh using the shader, which in turn will use our textures to render their content on the mesh
        //mesh.render(shader, GL20.GL_TRIANGLES);
        //Gdx.gl.glViewport(0, 0, rgbaTexture.getWidth(), rgbaTexture.getHeight());
        batch.draw(rgbaTexture, 0, 0, displayWidth, displayHeight/*,(int)fromX,(int)fromY,(int)(width-fromX),(int)(height-fromY),false,false*/);
        //Gdx.gl.glViewport(0, 0, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
        //shader.end();

        //batch.draw(rgbaTexture,0,Gdx.graphics.getHeight());

        isComputing=false;

    }

    public void renderBackground() {



        /*
         * Because of Java's limitations, we can't reference the middle of an array and
         * we must copy the channels in our byte array into buffers before setting them to textures
         */

        //Copy the Y channel of the image into its buffer, the first (width*height) bytes are the Y channel
        yBuffer.put(image, 0, width * height);
        yBuffer.position(0);

        //Copy the UV channels of the image into their buffer, the following (width*height/2) bytes are the UV channel; the U and V bytes are interspread
        uvBuffer.put(image, width * height, width * height / 2);
        uvBuffer.position(0);


        /*
         * Prepare the Y channel texture
         */

        //Set texture slot 0 as active and bind our texture object to it
        Gdx.gl.glActiveTexture(GL20.GL_TEXTURE0);
        yTexture.bind();

        //Y texture is (width*height) in size and each pixel is one byte; by setting GL_LUMINANCE, OpenGL puts this byte into R,G and B components of the texture
        Gdx.gl.glTexImage2D(GL20.GL_TEXTURE_2D, 0, GL20.GL_LUMINANCE, width, height, 0, GL20.GL_LUMINANCE, GL20.GL_UNSIGNED_BYTE, yBuffer);

        //Use linear interpolation when magnifying/minifying the texture to areas larger/smaller than the texture size
        Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_MIN_FILTER, GL20.GL_LINEAR);
        Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_MAG_FILTER, GL20.GL_LINEAR);
        Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_WRAP_S, GL20.GL_CLAMP_TO_EDGE);
        Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_WRAP_T, GL20.GL_CLAMP_TO_EDGE);


        /*
         * Prepare the UV channel texture
         */

        //Set texture slot 1 as active and bind our texture object to it
        Gdx.gl.glActiveTexture(GL20.GL_TEXTURE1);
        uvTexture.bind();

        //UV texture is (width/2*height/2) in size (downsampled by 2 in both dimensions, each pixel corresponds to 4 pixels of the Y channel)
        //and each pixel is two bytes. By setting GL_LUMINANCE_ALPHA, OpenGL puts first byte (V) into R,G and B components and of the texture
        //and the second byte (U) into the A component of the texture. That's why we find U and V at A and R respectively in the fragment shader code.
        //Note that we could have also found V at G or B as well.
        Gdx.gl.glTexImage2D(GL20.GL_TEXTURE_2D, 0, GL20.GL_LUMINANCE_ALPHA, width / 2, height / 2, 0, GL20.GL_LUMINANCE_ALPHA, GL20.GL_UNSIGNED_BYTE, uvBuffer);

        //Use linear interpolation when magnifying/minifying the texture to areas larger/smaller than the texture size
        Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_MIN_FILTER, GL20.GL_LINEAR);
        Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_MAG_FILTER, GL20.GL_LINEAR);
        Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_WRAP_S, GL20.GL_CLAMP_TO_EDGE);
        Gdx.gl.glTexParameterf(GL20.GL_TEXTURE_2D, GL20.GL_TEXTURE_WRAP_T, GL20.GL_CLAMP_TO_EDGE);

        /*
         * Draw the textures onto a mesh using our shader
         */

        shader.begin();

        //Set the uniform y_texture object to the texture at slot 0
        shader.setUniformi("y_texture", 0);

        //Set the uniform uv_texture object to the texture at slot 1
        shader.setUniformi("uv_texture", 1);

        //Render our mesh using the shader, which in turn will use our textures to render their content on the mesh
        mesh.render(shader, GL20.GL_TRIANGLES);
        SpriteBatch batch;
        //batch.draw(new Texture());
        shader.end();
    }

    public void destroy() {
        if (camera!=null) {
            camera.stopPreview();
            camera.setPreviewCallbackWithBuffer(null);
            camera.release();
        }
        if (camera2Api!=null){
            camera2Api.closeCamera();
        }
        frame.release();
        rgbaFrame.release();
        rgbaFrameRotated.release();

        mGrayScaled.release();
        points.release();
        faces.release();
        mGray.release();

    }
}
