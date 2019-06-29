//
// Created by Ali on 5/20/2019.
//

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/core/mat.hpp>
#include <android/log.h>
#include <jni.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "profiler.h"
#include "types.h"
#include "yuv2rgb.h"
#include "stdio.h"
#include "stdlib.h"


#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "dlib-jni:", __VA_ARGS__))

#define JNI_METHOD(NAME) \
    Java_ir_markazandroid_unimakeup_facedetect_FaceDetector_##NAME


void convertMatAddressToArray2d(JNIEnv *env,
                                long addrInputImage,
                                dlib::array2d<dlib::bgr_pixel> &out) {

    auto *pInputImage = (cv::Mat *) addrInputImage;

    //void* pixels;
    //int state;

    //LOGI("L%d: info.width=%d, info.height=%d", __LINE__, bitmapInfo.width, bitmapInfo.height);
    //out.set_size((long) pInputImage->rows, pInputImage->cols);

    //dlib::array2d<dlib::bgr_pixel> dlibImage;
    dlib::assign_image(out, dlib::cv_image<dlib::bgr_pixel>(*pInputImage));

    /* char* line = (char*) pixels;
     for (int h = 0; h < pInputImage->rows; ++h) {
         for (int w = 0; w < pInputImage->cols; ++w) {
             uint32_t* color = (uint32_t*) (line + 4 * w);

             out[h][w].red = (unsigned char) (0xFF & ((*color) >> 24));
             out[h][w].green = (unsigned char) (0xFF & ((*color) >> 16));
             out[h][w].blue = (unsigned char) (0xFF & ((*color) >> 8));
         }

         line = line + bitmapInfo.stride;
     }

     // Unlock the bitmap.
     AndroidBitmap_unlockPixels(env, bitmap);*/
}

void convertGrayMatAddressToArray2d(JNIEnv *env,
                                    long addrInputImage,
                                    dlib::array2d<unsigned char> &out) {

    auto *pInputImage = (cv::Mat *) addrInputImage;

    //void* pixels;
    //int state;

    //LOGI("L%d: info.width=%d, info.height=%d", __LINE__, bitmapInfo.width, bitmapInfo.height);
    //out.set_size((long) pInputImage->rows, pInputImage->cols);

    //dlib::array2d<dlib::bgr_pixel> dlibImage;

    /* char* line = (char*) pixels;
     for (int h = 0; h < pInputImage->rows; ++h) {
         for (int w = 0; w < pInputImage->cols; ++w) {
             uint32_t* color = (uint32_t*) (line + 4 * w);

             out[h][w].red = (unsigned char) (0xFF & ((*color) >> 24));
             out[h][w].green = (unsigned char) (0xFF & ((*color) >> 16));
             out[h][w].blue = (unsigned char) (0xFF & ((*color) >> 8));
         }

         line = line + bitmapInfo.stride;
     }

     // Unlock the bitmap.
     AndroidBitmap_unlockPixels(env, bitmap);*/

    dlib::assign_image(out, dlib::cv_image<unsigned char>(*pInputImage));
}

dlib::shape_predictor sFaceLandmarksDetector;
dlib::frontal_face_detector sFaceDetector;

extern "C" JNIEXPORT jboolean JNICALL
JNI_METHOD(isFaceDetectorReady)(JNIEnv *env,
                                jobject thiz) {
    if (sFaceDetector.num_detectors() > 0) {
        return JNI_TRUE;
    } else {
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
JNI_METHOD(isFaceLandmarksDetectorReady)(JNIEnv *env,
                                         jobject thiz) {
    if (sFaceLandmarksDetector.num_parts() > 0) {
        return JNI_TRUE;
    } else {
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT void JNICALL
JNI_METHOD(prepareFaceDetector)(JNIEnv *env,
                                jobject thiz) {

    // Prepare the detector.
    sFaceDetector = dlib::get_frontal_face_detector();

    // double interval = profiler.stopAndGetInterval();

    //LOGI("L%d: sFaceDetector is initialized (took %.3f ms)", __LINE__, interval);
    //LOGI("L%d: sFaceDetector.num_detectors()=%lu", __LINE__, sFaceDetector.num_detectors());
}

extern "C" JNIEXPORT void JNICALL
JNI_METHOD(prepareFaceLandmarksDetector)(JNIEnv *env,
                                         jobject thiz,
                                         jstring detectorPath) {
    const char *path = env->GetStringUTFChars(detectorPath, JNI_FALSE);

    // Profiler.
    //Profiler profiler;
    //profiler.start();

    // We need a shape_predictor. This is the tool that will predict face
    // landmark positions given an image and face bounding box.  Here we are just
    // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
    // as a command line argument.
    // Deserialize the shape detector.
    dlib::deserialize(path) >> sFaceLandmarksDetector;

    ///double interval = profiler.stopAndGetInterval();

    //LOGI("L%d: sFaceLandmarksDetector is initialized (took %.3f ms)", __LINE__, interval);
    //LOGI("L%d: sFaceLandmarksDetector.num_parts()=%lu", __LINE__, sFaceLandmarksDetector.num_parts());

    env->ReleaseStringUTFChars(detectorPath, path);

    if (sFaceLandmarksDetector.num_parts() != 68) {
        //throwException(env, "It's not a 68 landmarks detector!");
    }
}

using namespace cv;

extern "C" JNIEXPORT void JNICALL
JNI_METHOD(detectLandmark)(JNIEnv *env,
                           jobject thiz,
                           jlong matAddress,
                           jlong pointAddress) {
    if (sFaceDetector.num_detectors() == 0) {
        LOGI("L%d: sFaceDetector is not initialized!", __LINE__);
        //throwException(env, "sFaceDetector is not initialized!");
        // return NULL;
        return;
    }
    if (sFaceLandmarksDetector.num_parts() == 0) {
        LOGI("L%d: sFaceLandmarksDetector is not initialized!", __LINE__);
        //throwException(env, "sFaceLandmarksDetector is not initialized!");
        // return NULL;
        return;
    }

    // Profiler.
    Profiler profiler;
    profiler.start();
    // Convert bitmap to dlib::array2d.
    //dlib::array2d<dlib::bgr_pixel> img;
    //dlib::array2d<unsigned char> img;
    //convertGrayMatAddressToArray2d(env, matAddress, img);





//    // Make the image larger so we can detect small faces.
    //dlib::pyramid_up(img);
    //LOGI("L%d: pyramid_up the input image (w=%lu, h=%lu).", __LINE__, img.nc(), img.nr());

    auto *im = (cv::Mat *) matAddress;
    cv::Mat im_small, im_display;

    int resize = 4;

// Resize image for face detection
    cv::resize(*im, im_small, cv::Size(), 1.0 / resize, 1.0 / resize);




// Change to dlib's image format. No memory is copied.
    dlib::cv_image<unsigned char> cimg_small(im_small);
    dlib::cv_image<unsigned char> cimg(*im);


    const float width = (float) cimg_small.nc();
    const float height = (float) cimg_small.nr();
    double interval = profiler.stopAndGetInterval();
    LOGI("L%d: input image (w=%f, h=%f) is read (took %.3f ms)",
         __LINE__, width, height, interval);
    profiler.start();


    // Now tell the face detector to give us a list of bounding boxes
    // around all the faces in the image.
    std::vector<dlib::rectangle> dets = sFaceDetector(cimg_small, 0);
    interval = profiler.stopAndGetInterval();
    LOGI("L%d: Number of faces detected: %u (took %.3f ms)",
         __LINE__, (unsigned int) dets.size(), interval);

    // Protobuf message.
    //FaceList faces;
    // Now we will go ask the shape_predictor to tell us the pose of
    // each face we detected.
    if (!dets.empty()) {
        LOGI("L%d: face found OmG.", __LINE__);
        dlib::rectangle dets2 = dlib::rectangle(dets[0].left()*resize,dets[0].top()*resize,dets[0].right()*resize,dets[0].bottom()*resize);
        profiler.start();
        dlib::full_object_detection shape = sFaceLandmarksDetector(cimg, dets2);
        interval = profiler.stopAndGetInterval();
        LOGI("L%d: landmarks detected (took %.3f ms)",
             __LINE__, interval);

        auto *pointsMat = (cv::Mat *) pointAddress;

        for (unsigned int i = 0; i < 136; i = i + 2) {
            dlib::point &pt = shape.part(i / 2);
            //cv::Point2f point2f = cv::Point2f((float) pt.x(), (float) pt.y());

            pointsMat->at<float>(i, 0, 0) = (float) pt.x();
            pointsMat->at<float>(i + 1, 0, 0) = (float) pt.y();
        }
        pointsMat->at<float>(136, 0, 0) = (float) dets2.left();
        pointsMat->at<float>(137, 0, 0) = (float) dets2.top();
        pointsMat->at<float>(138, 0, 0) = (float) dets2.right();
        pointsMat->at<float>(139, 0, 0) = (float) dets2.bottom();

        // long *lp = (long*)malloc(sizeof(pointsMat));
        // return lp;
    } else
        LOGI("L%d: no faces found.", __LINE__);

    //return 0;
}

std::mutex _mutex;

extern "C" JNIEXPORT void JNICALL
JNI_METHOD(detectPoints)(JNIEnv *env,
                         jobject thiz,
                         jlong matAddress,
                         jlong pointAddress,
                         jlong tx,
                         jlong ty,
                         jlong bx,
                         jlong by) {


    Profiler profiler;

    if (sFaceDetector.num_detectors() == 0) {
        LOGI("L%d: sFaceDetector is not initialized!", __LINE__);
        //throwException(env, "sFaceDetector is not initialized!");
        // return NULL;
        return;
    }
    if (sFaceLandmarksDetector.num_parts() == 0) {
        LOGI("L%d: sFaceLandmarksDetector is not initialized!", __LINE__);
        //throwException(env, "sFaceLandmarksDetector is not initialized!");
        // return NULL;
        return;
    }

    //int res =5;

    dlib::rectangle det = dlib::rectangle(dlib::point(tx, ty), dlib::point(bx, by));

    // Profiler.


    Mat im = *((Mat *) matAddress);
    //Mat img = *im;
    //cv::Mat im_small;

// Resize image for face detection
    //cv::resize(*im, im_small, cv::Size(), 1.0/res, 1.0/res);

    profiler.start();
    //crop
    if (det.left() > 0 && det.right() < im.cols && det.top() > 0 && det.bottom() < im.rows) {
        cv::Rect faceROI(static_cast<int>(det.left()), static_cast<int>(det.top()),
                         static_cast<int>(det.width()), static_cast<int>(det.height()));
        cv::Mat face = im(faceROI);

        // apply filters
        cv::medianBlur(face, face, 5);  // remove noise
        // improve contrast
        cv::equalizeHist(face, face);
    }
    double interval = profiler.stopAndGetInterval();
    LOGI("L%d: crop time (took %.3f ms)",
         __LINE__, interval);

    dlib::cv_image<unsigned char> cimg(im);

    profiler.start();
    _mutex.lock();
    dlib::full_object_detection shape = sFaceLandmarksDetector(cimg, det);
    _mutex.unlock();

     interval = profiler.stopAndGetInterval();
    LOGI("L%d: point detected time (took %.3f ms)",
         __LINE__, interval);

    auto *pointsMat = (cv::Mat *) pointAddress;

    for (unsigned int i = 0; i < 136; i = i + 2) {
        dlib::point &pt = shape.part(i / 2);
        //cv::Point2f point2f = cv::Point2f((float) pt.x(), (float) pt.y());

        pointsMat->at<float>(i, 0, 0) = (float) pt.x();
        pointsMat->at<float>(i + 1, 0, 0) = (float) pt.y();
    }

}

using namespace imgUtils;

extern "C" JNIEXPORT void JNICALL JNI_METHOD(convertYUV420ToARGB8888)(
        JNIEnv* env, jclass clazz, jbyteArray y, jbyteArray u, jbyteArray v,
        jintArray output, jint width, jint height, jint y_row_stride,
        jint uv_row_stride, jint uv_pixel_stride, jboolean halfSize) {


    jboolean inputCopy = JNI_FALSE;
    jbyte* const y_buff = env->GetByteArrayElements(y, &inputCopy);
    jboolean outputCopy = JNI_FALSE;
    jint* const o = env->GetIntArrayElements(output, &outputCopy);


    if (halfSize) {
        ConvertYUV420SPToARGB8888HalfSize(reinterpret_cast<uint8*>(y_buff),
                                          reinterpret_cast<uint32*>(o), width,
                                          height);
    } else {
        jbyte* const u_buff = env->GetByteArrayElements(u, &inputCopy);
        jbyte* const v_buff = env->GetByteArrayElements(v, &inputCopy);

        ConvertYUV420ToARGB8888(
                reinterpret_cast<uint8*>(y_buff), reinterpret_cast<uint8*>(u_buff),
                reinterpret_cast<uint8*>(v_buff), reinterpret_cast<uint8*>(o), width,
                height, y_row_stride, uv_row_stride, uv_pixel_stride);

        env->ReleaseByteArrayElements(u, u_buff, JNI_ABORT);
        env->ReleaseByteArrayElements(v, v_buff, JNI_ABORT);
    }

    env->ReleaseByteArrayElements(y, y_buff, JNI_ABORT);
    env->ReleaseIntArrayElements(output, o, 0);
}

extern "C" JNIEXPORT void JNICALL JNI_METHOD(cvtYUV420ToARGB8888)(
        JNIEnv* env, jclass clazz, long inMatAddress,
        long outMatAddress, jint width, jint height, jint y_row_stride,
        jint uv_row_stride, jint uv_pixel_stride, jboolean halfSize) {


    //jboolean inputCopy = JNI_FALSE;
    //jbyte* const y_buff = env->GetByteArrayElements(y, &inputCopy);
    //jboolean outputCopy = JNI_FALSE;
    //jint* const o = env->GetIntArrayElements(output, &outputCopy);
    auto *inMat = (cv::Mat *) inMatAddress;
    auto *outMat = (cv::Mat *) outMatAddress;


    if (halfSize) {
        ConvertYUV420SPToARGB8888HalfSize(reinterpret_cast<uint8*>(y_buff),
                                          reinterpret_cast<uint32*>(o), width,
                                          height);
    } else {
        jbyte* const u_buff = env->GetByteArrayElements(u, &inputCopy);
        jbyte* const v_buff = env->GetByteArrayElements(v, &inputCopy);

        ConvertYUV420ToARGB8888(
                reinterpret_cast<uint8*>(y_buff), reinterpret_cast<uint8*>(u_buff),
                reinterpret_cast<uint8*>(v_buff), reinterpret_cast<uint8*>(o), width,
                height, y_row_stride, uv_row_stride, uv_pixel_stride);

        env->ReleaseByteArrayElements(u, u_buff, JNI_ABORT);
        env->ReleaseByteArrayElements(v, v_buff, JNI_ABORT);
    }

    env->ReleaseByteArrayElements(y, y_buff, JNI_ABORT);
    env->ReleaseIntArrayElements(output, o, 0);
}
