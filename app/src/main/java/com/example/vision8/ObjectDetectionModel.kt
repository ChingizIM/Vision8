package com.example.vision8

import android.content.Context
import android.graphics.Bitmap
import android.media.FaceDetector.Face.CONFIDENCE_THRESHOLD
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ObjectDetectionModel(context: Context) {
    private lateinit var interpreter: Interpreter

    init {
        // Load the model file
        val modelFile = context.assets.openFd("model.tflite")
        val inputStream = FileInputStream(modelFile.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = modelFile.startOffset
        val declaredLength = modelFile.declaredLength
        val mappedByteBuffer: MappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

        // Create the TensorFlow Lite interpreter
        val options = Interpreter.Options()
        interpreter = Interpreter(mappedByteBuffer, options)
    }

    fun detectObjects(bitmap: Bitmap): List<DetectionResult> {

        // Preprocess the input bitmap as necessary
        val inputArray = preprocessBitmap(bitmap)

        // Run the inference using the TensorFlow Lite interpreter

        val outputArray = Array(1) { Array(OUTPUT_COUNT) { FloatArray(4) } }
        fun Interpreter.runForMultipleInputsOutputs(inputs: Array<Any>,outputs: MutableMap<Int,Any>){
            runForMultipleInputsOutputs(inputs,outputs)
        }

        // Process the output array to generate the list of detection results
        return postprocessOutput(outputArray)
    }

    private fun preprocessBitmap(bitmap: Bitmap): Array<Array<FloatArray>> {
        // TODO: Implement the necessary preprocessing steps for your object detection model
        // Here's an example of how you might normalize the pixel values of a bitmap:
        val inputArray = Array(1) { Array(INPUT_SIZE) { FloatArray(INPUT_SIZE) } }
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = bitmap.getPixel(x, y)
                inputArray[0][x][y] = (pixel and 0xff) / 255.0f
            }
        }
        return inputArray
    }

    private fun postprocessOutput(outputArray: Array<Array<FloatArray>>): List<DetectionResult> {
        // TODO: Implement the necessary postprocessing steps for your object detection model
        // Here's an example of how you might extract the detected objects and their bounding boxes:
        val results = mutableListOf<DetectionResult>()
        for (i in 0 until OUTPUT_COUNT) {
            val confidence = outputArray[0][i][1]
            if (confidence >= CONFIDENCE_THRESHOLD) {
                val x1 = outputArray[0][i][0]
                val y1 = outputArray[0][i][2]
                val x2 = outputArray[0][i][3]
                val y2 = outputArray[0][i][4]
                results.add(DetectionResult(x1, y1, x2, y2, confidence))
            }
        }
        return results
    }

    companion object {
        const val INPUT_SIZE = 300
        const val OUTPUT_COUNT = 100
        const val CONFIDENCE_THRESHOLD = 0.5f

        fun newInstance(context: Context): ObjectDetectionModel {
            val model = loadModelFile(context)
            val options = Interpreter.Options().apply {
                setNumThreads(NUM_THREADS)
            }
            val interpreter = Interpreter(model, options)
            return ObjectDetectionModel(interpreter)
        }
        private fun loadModelFile(context: Context): MappedByteBuffer {
            val assetFileDescriptor = context.assets.openFd(MODEL_FILENAME)
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }

        private const val MODEL_FILENAME = "model.tflite"
        private const val NUM_THREADS = 4
    }

}