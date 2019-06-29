package ir.markazandroid.unimakeup.facedetect;

import org.opencv.core.Mat;

/**
 * Coded by Ali on 5/21/2019.
 */
public class FaceDetector {

    static {System.loadLibrary("native-lib");
        System.loadLibrary("c++_shared");}

    public native boolean isFaceDetectorReady();

    public native boolean isFaceLandmarksDetectorReady();

    public native void prepareFaceDetector();

    public native void prepareFaceLandmarksDetector(String path);

    public native void detectLandmark(long matAddress,long pointAddress);

    public native void detectPoints(long matAddress,long pointAddress,long tx,long ty,long bx,long by);

    /**
     * Converts YUV420 semi-planar data to ARGB 8888 data using the supplied width
     * and height. The input and output must already be allocated and non-null.
     * For efficiency, no error checking is performed.
     *
     * @param y
     * @param u
     * @param v
     * @param uvPixelStride
     * @param width         The width of the input image.
     * @param height        The height of the input image.
     * @param halfSize      If true, downsample to 50% in each dimension, otherwise not.
     * @param output        A pre-allocated array for the ARGB 8:8:8:8 output data.
     */
    public static native void convertYUV420ToARGB8888(
            byte[] y,
            byte[] u,
            byte[] v,
            int[] output,
            int width,
            int height,
            int yRowStride,
            int uvRowStride,
            int uvPixelStride,
            boolean halfSize);
}
