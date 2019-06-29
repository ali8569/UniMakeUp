package ir.markazandroid.unimakeup;

import android.os.Bundle;

import com.badlogic.gdx.backends.android.AndroidApplication;
import com.badlogic.gdx.backends.android.AndroidApplicationConfiguration;

public class CameraViewActivity extends AndroidApplication {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        AndroidApplicationConfiguration cfg = new AndroidApplicationConfiguration();
        cfg.a = 8;
        cfg.b = 8;
        cfg.g = 8;
        cfg.r = 8;

        AndroidDependentCameraController cameraControl = new AndroidDependentCameraController(this);
        initialize(new CameraRenderer(cameraControl), cfg);

        graphics.setContinuousRendering(false);
        graphics.getView().setKeepScreenOn(true);
    }
}
