package ai.ainamaura.checkers

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.Locale
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sqrt

class VoiceController(
    private val context: Context,
    private val coroutineScope: CoroutineScope,
    private val onAudioFeatures: (DoubleArray) -> Unit,
    private val onSpeechError: (String) -> Unit
) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = null
    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null
    var isTtsReady = false

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioEncoding = AudioFormat.ENCODING_PCM_16BIT
    // 100ms frame at 16kHz mono 16-bit = 1600 samples
    private val frameSamples = sampleRate / 10
    private val frameBytes = frameSamples * 2 // 16-bit = 2 bytes per sample

    init {
        tts = TextToSpeech(context, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                // TTS Language not supported
            } else {
                isTtsReady = true
                tts?.setSpeechRate(0.9f) // Slightly slower, more thoughtful
                tts?.setPitch(0.95f) // Slightly deeper for Ainamaura
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) {}
                    override fun onError(utteranceId: String?) {}
                })
            }
        }
    }

    fun speak(text: String, utteranceId: String = "AinamauraSpeech") {
        if (isTtsReady) {
            val cleanText = text.replace(Regex("\\[.*?\\]"), "").trim() // Remove telemetry tags like [Mapped Continuous-Time...] for speech
            val speechText = if (cleanText.startsWith("Ai:")) cleanText.substring(3).trim() else cleanText
            tts?.speak(speechText, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
        }
    }

    fun stopSpeaking() {
        if (isTtsReady) {
            tts?.stop()
        }
    }

    fun startListening() {
        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioEncoding)
        val bufferSize = maxOf(minBufferSize, frameBytes * 2)

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioEncoding,
                bufferSize
            )
        } catch (e: SecurityException) {
            onSpeechError("Microphone permission not granted: ${e.message}")
            return
        }

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            onSpeechError("AudioRecord failed to initialize")
            audioRecord?.release()
            audioRecord = null
            return
        }

        audioRecord?.startRecording()

        recordingJob = coroutineScope.launch(Dispatchers.IO) {
            val buffer = ShortArray(frameSamples)
            while (isActive) {
                val read = audioRecord?.read(buffer, 0, frameSamples) ?: break
                if (read == frameSamples) {
                    val features = extractFeatures(buffer, read)
                    onAudioFeatures(features)
                } else if (read < 0) {
                    break
                }
            }
        }
    }

    fun stopListening() {
        recordingJob?.cancel()
        recordingJob = null
        try {
            audioRecord?.stop()
        } catch (_: IllegalStateException) {
            // Already stopped
        }
        audioRecord?.release()
        audioRecord = null
    }

    fun release() {
        tts?.stop()
        tts?.shutdown()
        stopListening()
    }

    /**
     * Extracts 10 audio features from a 100ms PCM frame:
     *   [0] rms_energy
     *   [1] zero_crossing_rate
     *   [2] spectral_centroid_approx
     *   [3-9] 7 MFCC-style log energy bands
     */
    private fun extractFeatures(samples: ShortArray, count: Int): DoubleArray {
        val features = DoubleArray(10)

        // --- [0] RMS Energy ---
        var sumSquares = 0.0
        for (i in 0 until count) {
            val s = samples[i].toDouble()
            sumSquares += s * s
        }
        features[0] = sqrt(sumSquares / count)

        // --- [1] Zero Crossing Rate ---
        var zeroCrossings = 0
        for (i in 1 until count) {
            if ((samples[i] >= 0 && samples[i - 1] < 0) ||
                (samples[i] < 0 && samples[i - 1] >= 0)
            ) {
                zeroCrossings++
            }
        }
        features[1] = zeroCrossings.toDouble() / (count - 1)

        // --- FFT (simple radix-2 DFT on power-of-2 length) ---
        // Use the largest power of 2 <= count for a simple FFT
        val fftSize = Integer.highestOneBit(count)
        val real = DoubleArray(fftSize)
        val imag = DoubleArray(fftSize)
        for (i in 0 until fftSize) {
            real[i] = samples[i].toDouble()
        }
        fft(real, imag)

        val halfFft = fftSize / 2
        val magnitudes = DoubleArray(halfFft)
        for (i in 0 until halfFft) {
            magnitudes[i] = sqrt(real[i] * real[i] + imag[i] * imag[i])
        }

        // --- [2] Spectral Centroid Approx ---
        var weightedSum = 0.0
        var magSum = 0.0
        for (i in 0 until halfFft) {
            weightedSum += i * magnitudes[i]
            magSum += magnitudes[i]
        }
        features[2] = if (magSum > 0.0) weightedSum / magSum else 0.0

        // --- [3-9] 7 MFCC-style log energy bands ---
        // Divide the spectrum into 7 bands and compute log energy of each
        val numBands = 7
        val bandSize = halfFft / numBands
        for (band in 0 until numBands) {
            val start = band * bandSize
            val end = if (band == numBands - 1) halfFft else (band + 1) * bandSize
            var bandEnergy = 0.0
            for (i in start until end) {
                bandEnergy += magnitudes[i] * magnitudes[i]
            }
            // Log energy with floor to avoid log(0)
            features[3 + band] = ln(bandEnergy + 1e-10)
        }

        return features
    }

    /**
     * In-place iterative Cooley-Tukey radix-2 FFT.
     * Arrays must be power-of-2 length.
     */
    private fun fft(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        // Bit-reversal permutation
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                var temp = real[i]; real[i] = real[j]; real[j] = temp
                temp = imag[i]; imag[i] = imag[j]; imag[j] = temp
            }
        }
        // FFT butterfly
        var len = 2
        while (len <= n) {
            val halfLen = len / 2
            val angle = -2.0 * Math.PI / len
            val wReal = kotlin.math.cos(angle)
            val wImag = kotlin.math.sin(angle)
            var i = 0
            while (i < n) {
                var curReal = 1.0
                var curImag = 0.0
                for (k in 0 until halfLen) {
                    val tReal = curReal * real[i + k + halfLen] - curImag * imag[i + k + halfLen]
                    val tImag = curReal * imag[i + k + halfLen] + curImag * real[i + k + halfLen]
                    real[i + k + halfLen] = real[i + k] - tReal
                    imag[i + k + halfLen] = imag[i + k] - tImag
                    real[i + k] += tReal
                    imag[i + k] += tImag
                    val newCurReal = curReal * wReal - curImag * wImag
                    curImag = curReal * wImag + curImag * wReal
                    curReal = newCurReal
                }
                i += len
            }
            len = len shl 1
        }
    }
}
