package com.example.vision8

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.Image
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.objectdetection.ObjectDetectionModel
import com.google.common.util.concurrent.ListenableFuture
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import org.tensorflow.lite.support.image.ops.*
import org.tensorflow.lite.support.image.resize.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.common.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.SubtractMeanOp
import org.tensorflow.lite.support.common.ops.TensorOperator
import org.tensorflow.lite.support.common.ops.TransposeOp
import com.example.myapp.DetectionResult


class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var takePictureButton: Button
    private lateinit var resultsTextView: TextView

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var imageCapture: ImageCapture
    private lateinit var objectDetector: ObjectDetector

    private val objectDetectionModel = ObjectDetectionModel.newInstance(this)

    companion object {
        private const val TAG = "ObjectDetectionDemo"
        private const val THRESHOLD = 0.5f
    }

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.texture_view)
        takePictureButton = findViewById(R.id.photo_button)
        resultsTextView = findViewById(R.id.resultsTextView)

        takePictureButton.setOnClickListener {
            takePhoto()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun takePhoto() {
        // Configure CameraX to use the rear camera and display the image on the TextureView
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }
            val imageCapture = ImageCapture.Builder().build()

            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }

            // Take a picture and pass it to the TensorFlow Lite Object Detection API for object recognition
            imageCapture.takePicture(
                ContextCompat.getMainExecutor(this),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        // Convert the image to a bitmap
                        val bitmap = imageProxyToBitmap(image)

                        // Use TensorFlow Lite Object Detection API to detect objects in the photo
                        val objects = detectObjects(bitmap)

                        // Display information about the detected objects in the application
                        displayObjects(objects)

                        image.close()
                    }

                    override fun onError(exception: ImageCaptureException) {
                        Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                    }
                })
        }, ContextCompat.getMainExecutor(this))
    }


    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun detectObjects(bitmap: Bitmap): List<Recognition> {
        val inputImage = TensorImage(Element.DataType.FLOAT32)
        inputImage.load(bitmap)

        // Runs the object detection model on the input image.
        val outputs = model.process(inputImage.tensorBuffer)

        // Gets the output probabilities and labels and finds the top-k object classes.
        val scores = outputs.outputFeature0AsTensorBuffer
        val labels = outputs.outputFeature1AsTensorBuffer
        val recognitions = ArrayList<Recognition>()
        for (i in 0 until scores.size) {
            val score = scores.getFloat(i)
            if (score > THRESHOLD) {
                val label = labels.getString(i)
                val confidence = score
                val recognition = Recognition(label, confidence)
                recognitions.add(recognition)
            }
        }
        return recognitions
    }


    private fun displayObjects(objects: List<DetectionResult>) {
        val sb = StringBuilder()
        for (obj in objects) {
            sb.append("${obj.title}: ${obj.score}\n")
        }
        resultsTextView.text = sb.toString()
    }

    companion object {
        private const val TAG = "ObjectDetectionDemo"
    }
}