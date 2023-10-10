/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: PoseLandmarkerResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    init {
        initPaints()
    }

    fun clear() {
        results = null
        pointPaint.reset()
        linePaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color = Color.YELLOW
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.FILL_AND_STROKE

        pointPaint.color = Color.RED
        pointPaint.strokeWidth = LANDMARK_POINT_WIDTH
        pointPaint.style = Paint.Style.FILL_AND_STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        results?.let { poseLandmarkerResult ->
            for(landmark in poseLandmarkerResult.landmarks()) {
                var postLandmark = landmark
                var left = landmark[25]
                var right = landmark[26]

                var isLeft = left.z() < 0 && right.z() > 0
                var isRight = left.z() > 0 && right.z() < 0
                var isfull = left.z() * right.z() > 0

                if(isLeft)
                {
                    postLandmark = landmark.filterIndexed { index, _ -> index in listOf(27,25,23,11,13,15) }
                    renderLeft(canvas,landmark)
                }

                if(isRight)
                {
                    postLandmark = landmark.filterIndexed { index, _ -> index in listOf(28,26,24,12,14,16) }
                    renderRight(canvas,landmark)
                }

                if(isfull)
                {
                    postLandmark = landmark.filterIndexed { index, _ -> index in listOf(27,25,23,11,13,15,28,26,24,12,14,16) }
                    renderFull(canvas,landmark)
                }


                for(normalizedLandmark in postLandmark) {
                    canvas.drawCircle(
                        normalizedLandmark.x() * imageWidth * scaleFactor,
                        normalizedLandmark.y() * imageHeight * scaleFactor,
                        3F,
                        pointPaint
                    )
                }
            }
        }
    }

    private fun renderLeft (canvas:Canvas,landmark: List<NormalizedLandmark>) {
        canvas.drawLine(
            landmark[27].x() * imageWidth * scaleFactor,
            landmark[27].y() * imageHeight * scaleFactor,
            landmark[25].x() * imageWidth * scaleFactor,
            landmark[25].y() * imageHeight * scaleFactor,
            linePaint
        )

        canvas.drawLine(
            landmark[25].x() * imageWidth * scaleFactor,
            landmark[25].y() * imageHeight * scaleFactor,
            landmark[23].x() * imageWidth * scaleFactor,
            landmark[23].y() * imageHeight * scaleFactor,
            linePaint
        )

        canvas.drawLine(
            landmark[23].x() * imageWidth * scaleFactor,
            landmark[23].y() * imageHeight * scaleFactor,
            landmark[11].x() * imageWidth * scaleFactor,
            landmark[11].y() * imageHeight * scaleFactor,
            linePaint
        )

        canvas.drawLine(
            landmark[11].x() * imageWidth * scaleFactor,
            landmark[11].y() * imageHeight * scaleFactor,
            landmark[13].x() * imageWidth * scaleFactor,
            landmark[13].y() * imageHeight * scaleFactor,
            linePaint
        )

        canvas.drawLine(
            landmark[13].x() * imageWidth * scaleFactor,
            landmark[13].y() * imageHeight * scaleFactor,
            landmark[15].x() * imageWidth * scaleFactor,
            landmark[15].y() * imageHeight * scaleFactor,
            linePaint
        )

        renderArc(canvas,Pair(landmark[23].x() * imageWidth * scaleFactor, landmark[23].y() * imageHeight * scaleFactor),Pair(landmark[25].x() * imageWidth * scaleFactor, landmark[25].y() * imageHeight * scaleFactor)
            ,Pair(landmark[27].x() * imageWidth * scaleFactor, landmark[27].y() * imageHeight * scaleFactor))

        renderArc(canvas,Pair(landmark[25].x() * imageWidth * scaleFactor, landmark[25].y() * imageHeight * scaleFactor),Pair(landmark[23].x() * imageWidth * scaleFactor, landmark[23].y() * imageHeight * scaleFactor)
            ,Pair(landmark[11].x() * imageWidth * scaleFactor, landmark[11].y() * imageHeight * scaleFactor))

        renderArc(canvas,Pair(landmark[23].x() * imageWidth * scaleFactor, landmark[23].y() * imageHeight * scaleFactor),Pair(landmark[11].x() * imageWidth * scaleFactor, landmark[11].y() * imageHeight * scaleFactor)
            ,Pair(landmark[13].x() * imageWidth * scaleFactor, landmark[13].y() * imageHeight * scaleFactor))

        renderArc(canvas,Pair(landmark[15].x() * imageWidth * scaleFactor, landmark[15].y() * imageHeight * scaleFactor),Pair(landmark[13].x() * imageWidth * scaleFactor, landmark[13].y() * imageHeight * scaleFactor)
            ,Pair(landmark[11].x() * imageWidth * scaleFactor, landmark[11].y() * imageHeight * scaleFactor))
    }

    private fun renderArc(canvas:Canvas,startLine: Pair<Float, Float>, intersection: Pair<Float, Float>, endLine: Pair<Float, Float>) {
        val paint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 8f
        }

        val centerX = intersection.first
        val centerY = intersection.second
        val radius = 12F

        val startAngle = calculateAngle(startLine.first, startLine.second, centerX, centerY) // 圆弧的起始角度
        val endAngle = calculateAngle(endLine.first, endLine.second, centerX, centerY) // 圆弧的结束角度
        val rect = RectF(centerX - radius, centerY - radius, centerX + radius, centerY + radius)

        val path = Path()
        var sweepAngle = endAngle - startAngle
        if(sweepAngle < 0 )
        {
            sweepAngle += 360f;
        }
        path.arcTo(rect, startAngle, sweepAngle, true)

        canvas.drawPath(path, paint)

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 18f
            textAlign = Paint.Align.LEFT
        }
        canvas.drawText("${round(sweepAngle)}",centerX+3f,centerY+18f,textPaint)
    }

    private  fun calculateAngle(x1: Float, y1: Float, centerX: Float, centerY: Float): Float {
        val deltaX = x1 - centerX
        val deltaY = y1 - centerY
        val radians = atan2(deltaY.toDouble(), deltaX.toDouble())
        val degrees = Math.toDegrees(radians.toDouble()).toFloat()
        return if (degrees < 0) degrees + 360f else degrees
    }

    private fun renderRight (canvas:Canvas,landmark: List<NormalizedLandmark>) {
        canvas.drawLine(
            landmark[28].x() * imageWidth * scaleFactor,
            landmark[28].y() * imageHeight * scaleFactor,
            landmark[26].x() * imageWidth * scaleFactor,
            landmark[26].y() * imageHeight * scaleFactor,
            linePaint
        )

        canvas.drawLine(
            landmark[26].x() * imageWidth * scaleFactor,
            landmark[26].y() * imageHeight * scaleFactor,
            landmark[24].x() * imageWidth * scaleFactor,
            landmark[24].y() * imageHeight * scaleFactor,
            linePaint
        )

        canvas.drawLine(
            landmark[24].x() * imageWidth * scaleFactor,
            landmark[24].y() * imageHeight * scaleFactor,
            landmark[12].x() * imageWidth * scaleFactor,
            landmark[12].y() * imageHeight * scaleFactor,
            linePaint
        )

        canvas.drawLine(
            landmark[12].x() * imageWidth * scaleFactor,
            landmark[12].y() * imageHeight * scaleFactor,
            landmark[14].x() * imageWidth * scaleFactor,
            landmark[14].y() * imageHeight * scaleFactor,
            linePaint
        )

        canvas.drawLine(
            landmark[14].x() * imageWidth * scaleFactor,
            landmark[14].y() * imageHeight * scaleFactor,
            landmark[16].x() * imageWidth * scaleFactor,
            landmark[16].y() * imageHeight * scaleFactor,
            linePaint
        )

        renderArc(canvas,Pair(landmark[24].x() * imageWidth * scaleFactor, landmark[24].y() * imageHeight * scaleFactor),Pair(landmark[26].x() * imageWidth * scaleFactor, landmark[26].y() * imageHeight * scaleFactor)
            ,Pair(landmark[28].x() * imageWidth * scaleFactor, landmark[28].y() * imageHeight * scaleFactor))

        renderArc(canvas,Pair(landmark[26].x() * imageWidth * scaleFactor, landmark[26].y() * imageHeight * scaleFactor),Pair(landmark[24].x() * imageWidth * scaleFactor, landmark[24].y() * imageHeight * scaleFactor)
            ,Pair(landmark[12].x() * imageWidth * scaleFactor, landmark[12].y() * imageHeight * scaleFactor))

        renderArc(canvas,Pair(landmark[24].x() * imageWidth * scaleFactor, landmark[24].y() * imageHeight * scaleFactor),Pair(landmark[12].x() * imageWidth * scaleFactor, landmark[12].y() * imageHeight * scaleFactor)
            ,Pair(landmark[14].x() * imageWidth * scaleFactor, landmark[14].y() * imageHeight * scaleFactor))

        renderArc(canvas,Pair(landmark[16].x() * imageWidth * scaleFactor, landmark[16].y() * imageHeight * scaleFactor),Pair(landmark[14].x() * imageWidth * scaleFactor, landmark[14].y() * imageHeight * scaleFactor)
            ,Pair(landmark[12].x() * imageWidth * scaleFactor, landmark[12].y() * imageHeight * scaleFactor))
    }

    private fun renderFull (canvas:Canvas,landmark: List<NormalizedLandmark>) {
        renderLeft(canvas, landmark)
        renderRight(canvas, landmark)

        canvas.drawLine(
            landmark[23].x() * imageWidth * scaleFactor,
            landmark[23].y() * imageHeight * scaleFactor,
            landmark[24].x() * imageWidth * scaleFactor,
            landmark[24].y() * imageHeight * scaleFactor,
            linePaint
        )

        canvas.drawLine(
            landmark[11].x() * imageWidth * scaleFactor,
            landmark[11].y() * imageHeight * scaleFactor,
            landmark[12].x() * imageWidth * scaleFactor,
            landmark[12].y() * imageHeight * scaleFactor,
            linePaint
        )
    }

    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = poseLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 6F
        private const val LANDMARK_POINT_WIDTH = 12F
    }
}