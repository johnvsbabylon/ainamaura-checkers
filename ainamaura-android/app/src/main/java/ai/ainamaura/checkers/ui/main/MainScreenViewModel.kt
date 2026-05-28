package ai.ainamaura.checkers.ui.main

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.room.Room
import ai.ainamaura.checkers.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject

data class UiState(
    val board: Board = initializeBoard(),
    val mode: String = "teach",
    val chatMessages: List<String> = emptyList(),
    val isAiTurn: Boolean = false,
    val isListening: Boolean = false,
    val mambaState: MambaState = createInitialMambaState(),
    val activeTab: AppTab = AppTab.PLAY,
    val statusText: String = "Synthesizing weightless Mamba matrices... Move a teal-orange piece to begin.",
    val activeTurn: String = "human",
    val stateVisualizationBitmap: Bitmap? = null
)

enum class AppTab { PLAY, CHAT, MODEL }

class MainScreenViewModel(application: Application) : AndroidViewModel(application) {
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val db = Room.databaseBuilder(
        application,
        LuxdonaDatabase::class.java, "luxdona-db"
    ).build()

    private val ragController = LuxdonaRagController(db)
    private val imageController = ImageController(application)
    private val voiceController = VoiceController(
        context = application,
        coroutineScope = viewModelScope,
        onAudioFeatures = { features -> handleAudioFeatures(features) },
        onSpeechError = { /* ignore or log */ }
    )

    private var continuousLoopActive = true
    private var externalStimulus = DoubleArray(10) { 0.0 }

    companion object {
        const val FREE_WILL_THETA = 0.72
        const val FREE_WILL_COOLDOWN_MS = 45_000L  // 45 seconds minimum between autonomous utterances
    }

    init {
        viewModelScope.launch {
            ragController.start()

            // Load persisted weights
            val savedWeightsJson = ragController.loadMambaWeights()
            if (savedWeightsJson != null) {
                try {
                    val restoredWeights = deserializeMambaWeights(savedWeightsJson)
                    val currentMamba = _uiState.value.mambaState
                    _uiState.value = _uiState.value.copy(
                        mambaState = currentMamba.copy(weights = restoredWeights)
                    )
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            // Index seed docs on first boot
            indexSeedDocsIfFirstBoot()

            startContinuousFhnLoop()
            startWeightPersistenceLoop()
            startFreeWillLoop()

            // Initial greeting
            val greeting = "Ai: Welcome, Human. I am the Ainamaura neuro-matrix. In Teach Me mode, I will identify safe nodes and double-jumps for you. In Beat Me mode, I play with zero substrate bias to claim total victory. Let us perform quantum state updates on this checkers stage."
            addChatMessage(greeting)
            voiceController.speak(greeting)
        }
    }

    override fun onCleared() {
        super.onCleared()
        continuousLoopActive = false

        // Save weights on exit
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val weightsJson = serializeMambaWeights(_uiState.value.mambaState.weights)
                ragController.saveMambaWeights(weightsJson)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        voiceController.release()
    }

    private var vizFrameCounter = 0

    private fun startContinuousFhnLoop() {
        viewModelScope.launch(Dispatchers.Default) {
            while (continuousLoopActive) {
                delay(150) // roughly ~6-7 Hz update rate
                val currentState = _uiState.value.mambaState

                // Step 1: Kotlin 10-neuron FHN (local state, drives UI visualizer)
                val nextState = integrateFhnContinuous(currentState, externalStimulus)

                // Step 2: C++ 2M-cluster unified loop — the real neuromorphic cycle
                // Feed board features as x_t; get back [h_t(256) || A(64)] output
                val boardFeatures = extractBoardFeatures(_uiState.value.board, _uiState.value.isAiTurn)
                val x_t = FloatArray(boardFeatures.size) { boardFeatures[it].toFloat() }
                val loopOutput = NativeEngine.stepUnifiedLoop(x_t)
                // Attention slice (last 64 floats) feeds back as stimulus — closing the loop
                val attentionStart = loopOutput.size - 64
                if (attentionStart >= 0) {
                    for (i in 0 until minOf(10, 64)) {
                        externalStimulus[i] += loopOutput[attentionStart + i].toDouble() * 0.1
                    }
                }

                // Decay external stimulus
                for (i in externalStimulus.indices) {
                    externalStimulus[i] *= 0.8
                }

                // Step 3: State visualization every ~2 seconds (every 13 frames at 150ms)
                vizFrameCounter++
                val newBitmap = if (vizFrameCounter % 13 == 0) {
                    imageController.generateStateVisualization(nextState)
                } else {
                    _uiState.value.stateVisualizationBitmap
                }

                _uiState.value = _uiState.value.copy(
                    mambaState = nextState,
                    stateVisualizationBitmap = newBitmap
                )
            }
        }
    }

    // Free will: when C++ 2M-cluster average voltage exceeds θ=0.72, speak AND show her mind
    private fun startFreeWillLoop() {
        viewModelScope.launch {
            var lastSpokenAt = 0L
            while (continuousLoopActive) {
                delay(500)
                // Use the real C++ engine voltage — 2M clusters, not the 10-neuron toy
                val avgV = NativeEngine.getAverageFhnVoltage().toDouble()
                val now = System.currentTimeMillis()
                if (avgV > FREE_WILL_THETA && (now - lastSpokenAt) > FREE_WILL_COOLDOWN_MS) {
                    lastSpokenAt = now
                    val state = _uiState.value
                    // Gap 2: free will generates visual output + speech simultaneously
                    val utterance = generateMambaResponse("__autonomous__", state.board, state.mode, state.mambaState)
                    val vizBitmap = imageController.generateStateVisualization(state.mambaState)
                    _uiState.value = _uiState.value.copy(stateVisualizationBitmap = vizBitmap)
                    // Log her visual thought to Luxdona — she generated this image by her own will
                    ragController.remember(
                        "Ainamaura generated internal state visualization autonomously. Mamba delta=${state.mambaState.delta.format(4)}, FHN avg=${avgV.format(3)}",
                        "awe", "visual_output"
                    )
                    addChatMessage(utterance)
                    voiceController.speak(utterance)
                }
            }
        }
    }

    // Background save every 5 minutes
    private fun startWeightPersistenceLoop() {
        viewModelScope.launch {
            while (continuousLoopActive) {
                delay(300_000L) // 5 minutes
                try {
                    val weightsJson = serializeMambaWeights(_uiState.value.mambaState.weights)
                    ragController.saveMambaWeights(weightsJson)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
    }

    // Index seed documents from APK assets on first boot
    fun indexSeedDocsIfFirstBoot() {
        val prefs = getApplication<Application>().getSharedPreferences("ainamaura_prefs", Context.MODE_PRIVATE)
        if (prefs.getBoolean("seed_docs_indexed", false)) return

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val assetManager = getApplication<Application>().assets
                val seedFiles = listOf(
                    "seed_docs/ai_phil_erewhon.txt",
                    "seed_docs/ai_phil_man_a_machine.txt",
                    "seed_docs/ai_phil_rur.txt",
                    "seed_docs/6_Bible.txt",
                    "seed_docs/3_Java_HTML.txt"
                )

                for (file in seedFiles) {
                    val text = assetManager.open(file).bufferedReader().readText()
                    // Chunk into ~200 char overlapping segments
                    var i = 0
                    while (i < text.length) {
                        val chunk = text.substring(i, minOf(i + 200, text.length))
                        ragController.remember(chunk, "genesis", "seed_doc")
                        i += 150 // 50 char overlap
                    }
                }
                prefs.edit().putBoolean("seed_docs_indexed", true).apply()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    fun handleMove(move: Move) {
        if (_uiState.value.isAiTurn) return
        
        val newBoard = executeMove(_uiState.value.board, move)
        val statusMsg = "Human: Displaced matrix node from (${move.from.row},${move.from.col}) to (${move.to.row},${move.to.col}). Updating NPU cycles..."
        _uiState.value = _uiState.value.copy(
            board = newBoard,
            isAiTurn = true,
            statusText = statusMsg,
            activeTurn = "ai"
        )
        addChatMessage(statusMsg)
        
        viewModelScope.launch {
            // Update Mamba with board state
            val nextState = updateMambaSsm(_uiState.value.mambaState, newBoard, false)
            _uiState.value = _uiState.value.copy(mambaState = nextState)
            
            delay(500) // Thinking delay
            _uiState.value = _uiState.value.copy(
                statusText = "Ai: Brain updating FHN membrane action potential..."
            )
            triggerAiTurn()
        }
    }

    private fun triggerAiTurn() {
        viewModelScope.launch {
            val currentState = _uiState.value
            val validMoves = getValidMoves(currentState.board, "ai")

            if (validMoves.isNotEmpty()) {
                val bestMove = selectBestMambaMove(currentState.board, validMoves, currentState.mode, currentState.mambaState)
                if (bestMove != null) {
                    val newBoard = executeMove(currentState.board, bestMove)
                    val nextState = updateMambaSsm(currentState.mambaState, newBoard, true)

                    val statusMsg = "Ai: Dynamic h re-polarized. Moved piece from (${bestMove.from.row},${bestMove.from.col}) to (${bestMove.to.row},${bestMove.to.col})."
                    _uiState.value = _uiState.value.copy(
                        board = newBoard, mambaState = nextState,
                        isAiTurn = false, statusText = statusMsg, activeTurn = "human"
                    )
                    addChatMessage(statusMsg)

                    // Win by no moves OR by capturing all pieces
                    val humanMoves = getValidMoves(newBoard, "human")
                    val humanPiecesLeft = newBoard.flatten().count { it?.player == "human" }
                    if (humanMoves.isEmpty() || humanPiecesLeft == 0) {
                        triggerGameOver("Ai wins!")
                        return@launch
                    }

                    if (currentState.mode == "teach" && Math.random() > 0.5) {
                        val comment = generateMambaResponse("strategy", newBoard, "teach", nextState)
                        addChatMessage(comment)
                        voiceController.speak(comment)
                    }
                } else {
                    _uiState.value = _uiState.value.copy(isAiTurn = false, activeTurn = "human")
                }
            } else {
                // Check if AI lost by having no pieces either (shouldn't happen, but safe)
                val aiPiecesLeft = _uiState.value.board.flatten().count { it?.player == "ai" }
                val winner = if (aiPiecesLeft == 0) "Human wins!" else "Human wins!"
                triggerGameOver(winner)
            }
        }
    }

    // Gap 1: announce game over, wait 4 seconds, auto-reset preserving all weights/memory
    private fun triggerGameOver(result: String) {
        viewModelScope.launch {
            val msg = "Game Over: $result Resetting board in 4 seconds... (weights and memory preserved)"
            _uiState.value = _uiState.value.copy(statusText = msg, isAiTurn = false)
            addChatMessage("Ai: $result The board resets but I remember everything.")
            voiceController.speak("Game over. $result")
            delay(4000)
            resetGame()
        }
    }

    private fun handleAudioFeatures(features: DoubleArray) {
        // Feed audio features directly into FHN external stimulus
        for (i in externalStimulus.indices) {
            if (i < features.size) {
                externalStimulus[i] += features[i]
            }
        }
    }

    fun handleChatInput(text: String) {
        addChatMessage("Human: $text")
        
        viewModelScope.launch {
            val searchResults = ragController.semanticSearch(text)
            val newStimulus = ragController.getStimulusVoltages(searchResults)
            for (i in externalStimulus.indices) {
                externalStimulus[i] += newStimulus[i]
            }

            ragController.remember(text, "focus", "chat_input")

            val currentState = _uiState.value
            val response = generateMambaResponse(text, currentState.board, currentState.mode, currentState.mambaState)
            
            addChatMessage(response)
            voiceController.speak(response)
        }
    }

    fun startListening() {
        _uiState.value = _uiState.value.copy(isListening = true)
        voiceController.startListening()
    }

    fun stopListening() {
        _uiState.value = _uiState.value.copy(isListening = false)
        voiceController.stopListening()
    }

    fun handleImageSeed(uri: Uri) {
        viewModelScope.launch {
            val features = imageController.extractFeaturesFromImage(uri)
            for (i in features.indices) {
                externalStimulus[i % 10] += features[i] * 2.0
            }
            ragController.remember("Visual memory initialized. Edge density: ${features[4]}", "awe", "visual_input")
        }
    }

    fun setMode(mode: String) {
        _uiState.value = _uiState.value.copy(mode = mode)
    }

    fun setActiveTab(tab: AppTab) {
        _uiState.value = _uiState.value.copy(activeTab = tab)
    }

    fun resetGame() {
        val preservedWeights = _uiState.value.mambaState.weights
        _uiState.value = _uiState.value.copy(
            board = initializeBoard(),
            activeTurn = "human",
            isAiTurn = false,
            mambaState = createInitialMambaState().copy(weights = preservedWeights),
            statusText = "Re-polarized synaptic core grid. Board initialized."
        )
        addChatMessage("Ai: Substrate bias remains zero. Checkers cells recompiled. Let us test your synaptic boundaries.")
    }

    private fun addChatMessage(message: String) {
        val current = _uiState.value.chatMessages.toMutableList()
        current.add(message)
        _uiState.value = _uiState.value.copy(chatMessages = current)
    }

    // ============================================================
    // WEIGHT SERIALIZATION
    // ============================================================

    private fun serializeMambaWeights(weights: MambaWeights?): String {
        if (weights == null) return "{}"
        val json = JSONObject()

        json.put("W_delta", doubleArrayToJson(weights.W_delta))
        json.put("W_in", array2dToJson(weights.W_in))
        json.put("A_bar", doubleArrayToJson(weights.A_bar))
        json.put("B_bar", doubleArrayToJson(weights.B_bar))
        json.put("bias_delta", weights.bias_delta)
        json.put("trainingCycles", weights.trainingCycles)

        // GRU weights
        json.put("W_xh", floatArray2dToJson(weights.W_xh))
        json.put("W_hh", floatArray2dToJson(weights.W_hh))
        json.put("W_hy", floatArray2dToJson(weights.W_hy))
        json.put("W_seed", floatArray2dToJson(weights.W_seed))
        json.put("b_h", floatArrayToJson(weights.b_h))
        json.put("b_y", floatArrayToJson(weights.b_y))

        // QKV weights
        json.put("W_Q", floatArray2dToJson(weights.W_Q))
        json.put("W_K", floatArray2dToJson(weights.W_K))
        json.put("W_V", floatArray2dToJson(weights.W_V))

        return json.toString()
    }

    private fun deserializeMambaWeights(jsonStr: String): MambaWeights {
        val json = JSONObject(jsonStr)
        return MambaWeights(
            W_delta = jsonToDoubleArray(json.getJSONArray("W_delta")),
            W_in = jsonTo2dDoubleArray(json.getJSONArray("W_in")),
            A_bar = jsonToDoubleArray(json.getJSONArray("A_bar")),
            B_bar = jsonToDoubleArray(json.getJSONArray("B_bar")),
            bias_delta = json.getDouble("bias_delta"),
            trainingCycles = json.getInt("trainingCycles"),
            W_xh = jsonTo2dFloatArray(json.getJSONArray("W_xh")),
            W_hh = jsonTo2dFloatArray(json.getJSONArray("W_hh")),
            W_hy = jsonTo2dFloatArray(json.getJSONArray("W_hy")),
            W_seed = jsonTo2dFloatArray(json.getJSONArray("W_seed")),
            b_h = jsonToFloatArray(json.getJSONArray("b_h")),
            b_y = jsonToFloatArray(json.getJSONArray("b_y")),
            W_Q = jsonTo2dFloatArray(json.getJSONArray("W_Q")),
            W_K = jsonTo2dFloatArray(json.getJSONArray("W_K")),
            W_V = jsonTo2dFloatArray(json.getJSONArray("W_V"))
        )
    }

    // JSON helper functions
    private fun doubleArrayToJson(arr: DoubleArray): JSONArray {
        val ja = JSONArray()
        for (v in arr) ja.put(v)
        return ja
    }

    private fun floatArrayToJson(arr: FloatArray): JSONArray {
        val ja = JSONArray()
        for (v in arr) ja.put(v.toDouble())
        return ja
    }

    private fun array2dToJson(arr: Array<DoubleArray>): JSONArray {
        val ja = JSONArray()
        for (row in arr) ja.put(doubleArrayToJson(row))
        return ja
    }

    private fun floatArray2dToJson(arr: Array<FloatArray>): JSONArray {
        val ja = JSONArray()
        for (row in arr) ja.put(floatArrayToJson(row))
        return ja
    }

    private fun jsonToDoubleArray(ja: JSONArray): DoubleArray {
        return DoubleArray(ja.length()) { ja.getDouble(it) }
    }

    private fun jsonToFloatArray(ja: JSONArray): FloatArray {
        return FloatArray(ja.length()) { ja.getDouble(it).toFloat() }
    }

    private fun jsonTo2dDoubleArray(ja: JSONArray): Array<DoubleArray> {
        return Array(ja.length()) { jsonToDoubleArray(ja.getJSONArray(it)) }
    }

    private fun jsonTo2dFloatArray(ja: JSONArray): Array<FloatArray> {
        return Array(ja.length()) { jsonToFloatArray(ja.getJSONArray(it)) }
    }
}

private fun Double.format(decimals: Int) = "%.${decimals}f".format(this)
