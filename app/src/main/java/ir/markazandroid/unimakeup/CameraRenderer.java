package ir.markazandroid.unimakeup;

import android.graphics.Canvas;
import android.graphics.Rect;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;

import com.badlogic.gdx.ApplicationListener;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.GL30;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.opengles.GL11;

/**
 * Coded by Ali on 5/26/2019.
 */
public class CameraRenderer implements ApplicationListener {

    private final AndroidDependentCameraController deviceCameraControl;

    private SpriteBatch spriteBatch;

    public CameraRenderer(AndroidDependentCameraController cameraControl) {
        this.deviceCameraControl = cameraControl;
    }

    @Override
    public void create() {
        deviceCameraControl.init();
        spriteBatch=new SpriteBatch();
        Log.e("Width height",Gdx.graphics.getWidth()+"---"+Gdx.graphics.getHeight());
    }

    @Override
    public void render() {
        Gdx.gl.glViewport(0, 0, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT);

        //Render the background that is the live camera image
        spriteBatch.begin();
        deviceCameraControl.drawRgba(spriteBatch);
        spriteBatch.end();

        Log.e("fps",""+Gdx.graphics.getFramesPerSecond());
        /*
         * Render anything here (sprites/models etc.) that you want to go on top of the camera image
         */
    }

    @Override
    public void dispose() {
        deviceCameraControl.destroy();
    }

    @Override
    public void resize(int width, int height) {
    }

    @Override
    public void pause() {
    }

    @Override
    public void resume() {
    }
}