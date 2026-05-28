package ai.ainamaura.checkers

object NativeEngine {
    init {
        System.loadLibrary("ainamaura_engine")
    }

    external fun initEngine()
    
    // Returns the combined output [h_t || A]
    external fun stepUnifiedLoop(x: FloatArray): FloatArray
    
    external fun getAverageFhnVoltage(): Float
    
    external fun embedChunk(chunkVector: FloatArray): FloatArray
}
