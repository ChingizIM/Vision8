package com.example.vision8

import android.graphics.RectF

class DetectionResult(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val confidence: Float) {
    override fun toString(): String {
        return "DetectionResult(x1=$x1, y1=$y1, x2=$x2, y2=$y2, confidence=$confidence)"
    }
}
