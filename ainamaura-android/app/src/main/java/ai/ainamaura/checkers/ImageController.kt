package ai.ainamaura.checkers

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color as AndroidColor
import android.graphics.ImageDecoder
import android.graphics.Paint
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import kotlin.math.sqrt

class ImageController(private val context: Context) {

    fun extractFeaturesFromImage(uri: Uri): DoubleArray {
        val features = DoubleArray(10) { 0.0 }
        
        try {
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(context.contentResolver, uri)
                ImageDecoder.decodeBitmap(source)
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
            }

            // Downscale to process quickly
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 64, 64, true)
            
            var rSum = 0L
            var gSum = 0L
            var bSum = 0L
            var brightnessSum = 0L
            var edgeDensity = 0L

            for (x in 0 until 64) {
                for (y in 0 until 64) {
                    val pixel = scaledBitmap.getPixel(x, y)
                    val r = (pixel shr 16) and 0xff
                    val g = (pixel shr 8) and 0xff
                    val b = pixel and 0xff
                    
                    rSum += r
                    gSum += g
                    bSum += b
                    
                    val brightness = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
                    brightnessSum += brightness

                    if (x > 0 && y > 0) {
                        val prevPixelX = scaledBitmap.getPixel(x - 1, y)
                        val prevBrightnessX = (0.299 * ((prevPixelX shr 16) and 0xff) + 0.587 * ((prevPixelX shr 8) and 0xff) + 0.114 * (prevPixelX and 0xff)).toInt()
                        
                        val prevPixelY = scaledBitmap.getPixel(x, y - 1)
                        val prevBrightnessY = (0.299 * ((prevPixelY shr 16) and 0xff) + 0.587 * ((prevPixelY shr 8) and 0xff) + 0.114 * (prevPixelY and 0xff)).toInt()

                        val edge = sqrt(((brightness - prevBrightnessX) * (brightness - prevBrightnessX) + (brightness - prevBrightnessY) * (brightness - prevBrightnessY)).toDouble())
                        if (edge > 20) {
                            edgeDensity++
                        }
                    }
                }
            }
            
            val totalPixels = 64 * 64
            
            features[0] = (rSum.toDouble() / totalPixels) / 255.0
            features[1] = (gSum.toDouble() / totalPixels) / 255.0
            features[2] = (bSum.toDouble() / totalPixels) / 255.0
            features[3] = (brightnessSum.toDouble() / totalPixels) / 255.0
            features[4] = edgeDensity.toDouble() / totalPixels
            
            // Random noise for remaining features, as placeholder for proper embeddings
            for (i in 5 until 10) {
                features[i] = Math.random() * 0.5
            }

        } catch (e: Exception) {
            e.printStackTrace()
        }
        
        return features
    }

    fun generateStateVisualization(mambaState: MambaState): Bitmap {
        val width = 256
        val height = 256
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        // Fill background with #030303
        canvas.drawColor(AndroidColor.argb(255, 3, 3, 3))

        val paint = Paint()

        // Draw FHN voltage as circles (10 clusters)
        val numClusters = 10
        for (i in 0 until numClusters) {
            val v = mambaState.fhnVoltage[i].toFloat()
            val x = (width * (i + 0.5f) / numClusters)
            val y = height / 2f
            val radius = (10f + v * 20f).coerceIn(2f, 40f)
            val alpha = ((v + 1f) / 2f * 200f).toInt().coerceIn(50, 255)
            paint.color = AndroidColor.argb(alpha, 59, 130, 246)
            paint.style = Paint.Style.FILL
            canvas.drawCircle(x, y, radius, paint)
        }

        // Draw Mamba h as waveform across bottom third
        val h = mambaState.h
        paint.color = AndroidColor.argb(180, 20, 184, 166)
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2f
        for (i in 1 until h.size) {
            val x1 = (width * (i - 1).toFloat() / h.size)
            val x2 = (width * i.toFloat() / h.size)
            val y1 = (height * 0.8f - h[i - 1].toFloat() * 30f)
            val y2 = (height * 0.8f - h[i].toFloat() * 30f)
            canvas.drawLine(x1, y1, x2, y2, paint)
        }

        return bitmap
    }
}
