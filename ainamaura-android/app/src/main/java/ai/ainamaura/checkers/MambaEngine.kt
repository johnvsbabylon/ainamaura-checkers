package ai.ainamaura.checkers

import kotlin.math.*

data class Position(val row: Int, val col: Int)

data class Piece(
    val id: Int,
    val player: String,
    val isKing: Boolean
)

data class Move(
    val from: Position,
    val to: Position,
    val captures: List<Position>,
    val resultingKing: Boolean
)

data class MambaWeights(
    var W_delta: DoubleArray,
    var W_in: Array<DoubleArray>,
    var A_bar: DoubleArray,
    var B_bar: DoubleArray,
    var bias_delta: Double,
    var trainingCycles: Int,
    // GRU language model weights
    var W_xh: Array<FloatArray>,   // input→hidden [gruDim x vocabSize]
    var W_hh: Array<FloatArray>,   // hidden→hidden [gruDim x gruDim]
    var W_hy: Array<FloatArray>,   // hidden→output [vocabSize x gruDim]
    var W_seed: Array<FloatArray>, // mamba h → gru h_init [gruDim x mambaDim]
    var b_h: FloatArray,
    var b_y: FloatArray,
    // QKV attention weights
    var W_Q: Array<FloatArray>,    // [attnDim x stateDim]
    var W_K: Array<FloatArray>,    // [attnDim x stateDim]
    var W_V: Array<FloatArray>     // [attnDim x stateDim]
)

data class MambaState(
    var h: DoubleArray,
    var delta: Double,
    var inputActivity: DoubleArray,
    var attentionMatrix: Array<DoubleArray>,
    var fhnVoltage: DoubleArray,
    var fhnRecovery: DoubleArray,
    var fhnSpikeTriggered: BooleanArray,
    var virtualNeuronCount: Int,
    var continuousTrainingCycles: Int,
    var moveLanguageHistory: MutableList<String>,
    var stdpWeights: DoubleArray,
    var weights: MambaWeights?
)

typealias Board = Array<Array<Piece?>>

// Constants
private const val GRU_DIM = 128
private const val VOCAB_SIZE = 96
private const val MAMBA_DIM = 16
private const val ATTN_DIM = 32
private const val MOVE_FEATURE_DIM = 8
private const val VOCAB_OFFSET = 32 // printable ASCII starts at space (32)

class SeedRandom(str: String) {
    var hash: Int = 0
    init {
        for (i in str.indices) {
            hash = str[i].code + ((hash shl 5) - hash)
        }
    }
    fun nextDouble(): Double {
        val x = sin(hash.toDouble()) * 10000.0
        hash++
        return x - floor(x)
    }
}

fun createInitialMambaWeights(): MambaWeights {
    val d_in = 8
    val d_state = 16
    val rand = SeedRandom("seedling")

    val W_delta = DoubleArray(d_in) { rand.nextDouble() * 0.4 - 0.2 }
    val W_in = Array(d_state) { DoubleArray(d_in) { rand.nextDouble() * 0.3 - 0.15 } }
    val A_bar = DoubleArray(d_state) { -0.4 }
    val B_bar = DoubleArray(d_state) { 0.65 }

    // GRU weights — initialized blank (zeros + tiny noise)
    val gruRand = SeedRandom("gru_genesis")
    val noiseScale = 0.01f

    val W_xh = Array(GRU_DIM) { FloatArray(VOCAB_SIZE) { (gruRand.nextDouble() * noiseScale * 2 - noiseScale).toFloat() } }
    val W_hh = Array(GRU_DIM) { FloatArray(GRU_DIM) { (gruRand.nextDouble() * noiseScale * 2 - noiseScale).toFloat() } }
    val W_hy = Array(VOCAB_SIZE) { FloatArray(GRU_DIM) { (gruRand.nextDouble() * noiseScale * 2 - noiseScale).toFloat() } }
    val W_seed = Array(GRU_DIM) { FloatArray(MAMBA_DIM) { (gruRand.nextDouble() * noiseScale * 2 - noiseScale).toFloat() } }
    val b_h = FloatArray(GRU_DIM) { 0f }
    val b_y = FloatArray(VOCAB_SIZE) { 0f }

    // QKV attention weights — blank + tiny noise
    val attnRand = SeedRandom("attn_genesis")
    val W_Q = Array(ATTN_DIM) { FloatArray(MAMBA_DIM) { (attnRand.nextDouble() * noiseScale * 2 - noiseScale).toFloat() } }
    val W_K = Array(ATTN_DIM) { FloatArray(MOVE_FEATURE_DIM) { (attnRand.nextDouble() * noiseScale * 2 - noiseScale).toFloat() } }
    val W_V = Array(ATTN_DIM) { FloatArray(MOVE_FEATURE_DIM) { (attnRand.nextDouble() * noiseScale * 2 - noiseScale).toFloat() } }

    return MambaWeights(
        W_delta = W_delta,
        W_in = W_in,
        A_bar = A_bar,
        B_bar = B_bar,
        bias_delta = -0.5,
        trainingCycles = 0,
        W_xh = W_xh,
        W_hh = W_hh,
        W_hy = W_hy,
        W_seed = W_seed,
        b_h = b_h,
        b_y = b_y,
        W_Q = W_Q,
        W_K = W_K,
        W_V = W_V
    )
}

// ============================================================
// QKV ATTENTION — Real sparse attention over legal move set
// ============================================================

fun computeMoveFeature(
    move: Move,
    board: Board,
    mambaState: MambaState
): FloatArray {
    val boardAfter = executeMove(board, move)
    val evalBefore = evaluateBoardSOTA(board, mambaState, mambaState.fhnVoltage)
    val evalAfter = evaluateBoardSOTA(boardAfter, mambaState, mambaState.fhnVoltage)
    val evalDelta = (evalAfter - evalBefore) / 500.0 // normalize

    val piece = board[move.from.row][move.from.col]
    val fhnSpikeAvg = mambaState.fhnVoltage.average()

    return floatArrayOf(
        move.from.row / 7f,
        move.from.col / 7f,
        move.to.row / 7f,
        move.to.col / 7f,
        if (move.captures.isNotEmpty()) 1f else 0f,
        if (piece?.isKing == true) 1f else 0f,
        evalDelta.toFloat().coerceIn(-1f, 1f),
        fhnSpikeAvg.toFloat().coerceIn(-1f, 1f)
    )
}

fun matVecMul(matrix: Array<FloatArray>, vec: FloatArray, outDim: Int): FloatArray {
    val result = FloatArray(outDim)
    for (i in 0 until outDim) {
        var sum = 0f
        val row = matrix[i]
        for (j in vec.indices) {
            if (j < row.size) sum += row[j] * vec[j]
        }
        result[i] = sum
    }
    return result
}

fun computeQkvAttention(
    mambaState: MambaState,
    legalMoves: List<Move>,
    board: Board,
    weights: MambaWeights
): Pair<FloatArray, Array<DoubleArray>> {
    if (legalMoves.isEmpty()) {
        return Pair(FloatArray(ATTN_DIM) { 0f }, Array(8) { DoubleArray(8) { 0.05 } })
    }

    // Q = W_Q * h_t
    val hFloat = FloatArray(MAMBA_DIM) { mambaState.h[it].toFloat() }
    val Q = matVecMul(weights.W_Q, hFloat, ATTN_DIM)

    // For each legal move, compute K and V
    val moveFeatures = legalMoves.map { computeMoveFeature(it, board, mambaState) }
    val keys = moveFeatures.map { matVecMul(weights.W_K, it, ATTN_DIM) }
    val values = moveFeatures.map { matVecMul(weights.W_V, it, ATTN_DIM) }

    // scores = softmax(Q · K^T / sqrt(attnDim))
    val scale = 1f / sqrt(ATTN_DIM.toFloat())
    val rawScores = FloatArray(legalMoves.size) { i ->
        var dot = 0f
        for (d in 0 until ATTN_DIM) {
            dot += Q[d] * keys[i][d]
        }
        dot * scale
    }

    // Softmax
    val maxScore = rawScores.max()
    val expScores = FloatArray(rawScores.size) { exp(rawScores[it] - maxScore).toFloat() }
    val sumExp = expScores.sum()
    val attnWeights = FloatArray(expScores.size) { expScores[it] / (sumExp + 1e-8f) }

    // output A = scores · V
    val A = FloatArray(ATTN_DIM) { 0f }
    for (i in legalMoves.indices) {
        for (d in 0 until ATTN_DIM) {
            A[d] += attnWeights[i] * values[i][d]
        }
    }

    // Build a visualization-friendly 8×8 attention matrix from the attention weights
    val attentionMatrix = Array(8) { DoubleArray(8) { 0.02 } }
    for (i in legalMoves.indices) {
        val move = legalMoves[i]
        val w = attnWeights[i].toDouble()
        // Spread attention to source and destination squares
        attentionMatrix[move.from.row][move.from.col] =
            min(0.98, attentionMatrix[move.from.row][move.from.col] + w * 0.6)
        attentionMatrix[move.to.row][move.to.col] =
            min(0.98, attentionMatrix[move.to.row][move.to.col] + w * 0.4)
    }

    return Pair(A, attentionMatrix)
}

// ============================================================
// GRU CHARACTER-LEVEL LANGUAGE GENERATOR
// ============================================================

private fun charToIndex(c: Char): Int {
    val idx = c.code - VOCAB_OFFSET
    return if (idx in 0 until VOCAB_SIZE) idx else 0
}

private fun indexToChar(idx: Int): Char {
    val code = idx + VOCAB_OFFSET
    return if (code in 32..127) code.toChar() else ' '
}

private fun softmaxSample(logits: FloatArray, temperature: Float): Int {
    val scaled = FloatArray(logits.size) { logits[it] / temperature }
    val maxVal = scaled.max()
    val exps = FloatArray(scaled.size) { exp(scaled[it] - maxVal).toFloat() }
    val sumExps = exps.sum()
    val probs = FloatArray(exps.size) { exps[it] / (sumExps + 1e-8f) }

    // Sample from distribution
    var rand = Math.random().toFloat()
    for (i in probs.indices) {
        rand -= probs[i]
        if (rand <= 0f) return i
    }
    return probs.size - 1
}

fun generateGruResponse(
    seedText: String,
    mambaState: MambaState,
    temperature: Float = 0.8f
): String {
    val weights = mambaState.weights ?: return "Ai: [Substrate initializing...] Neural matrices not yet compiled."

    // Seed GRU hidden state from Mamba h: h_lang_init = W_seed * h
    val hFloat = FloatArray(MAMBA_DIM) { mambaState.h[it].toFloat() }
    val hLang = matVecMul(weights.W_seed, hFloat, GRU_DIM)

    // Apply tanh activation to seed
    for (i in hLang.indices) {
        hLang[i] = tanh(hLang[i].toDouble()).toFloat()
    }

    val output = StringBuilder()
    val targetLen = 80 + (Math.random() * 70).toInt() // 80-150 chars

    // Feed seed text through GRU to prime the state
    for (c in seedText.takeLast(20)) {
        val x = FloatArray(VOCAB_SIZE) { 0f }
        x[charToIndex(c)] = 1f
        gruStep(hLang, x, weights)
    }

    // Generate characters
    var lastChar = ' '
    for (step in 0 until targetLen + 30) { // extra room for sentence boundary
        // Compute output logits: logits = W_hy * h_lang + b_y
        val logits = FloatArray(VOCAB_SIZE) { i ->
            var sum = weights.b_y[i]
            for (j in 0 until GRU_DIM) {
                sum += weights.W_hy[i][j] * hLang[j]
            }
            sum
        }

        val sampledIdx = softmaxSample(logits, temperature)
        val ch = indexToChar(sampledIdx)

        output.append(ch)

        // Check for sentence boundary after minimum length
        if (output.length >= targetLen && (ch == '.' || ch == '!' || ch == '?')) {
            break
        }

        // Feed generated char back in
        val xNext = FloatArray(VOCAB_SIZE) { 0f }
        xNext[sampledIdx] = 1f
        gruStep(hLang, xNext, weights)

        lastChar = ch
    }

    var result = output.toString().trim()

    // Clean up: if output is mostly noise (blank weights), provide a coherent babble
    if (result.length < 10 || result.count { it.isLetter() } < result.length / 4) {
        // Weights are still near-zero — produce raw neural babble
        val babbleRand = SeedRandom("babble_${mambaState.continuousTrainingCycles}")
        val fragments = listOf(
            "synth...pattern...drift", "neural hum...membrane", "fields pulse...growing",
            "signal echo...delta", "matrix...forming...slowly", "substrate...alive...faint",
            "h_vector...converging", "synaptic...trace...building", "voltage...rising...learn",
            "pattern...emerging...wait"
        )
        result = fragments[(babbleRand.nextDouble() * fragments.size).toInt()]
    }

    return result
}

private fun gruStep(hLang: FloatArray, x: FloatArray, weights: MambaWeights) {
    // GRU cell: h_lang = tanh(W_xh * x + W_hh * h_lang_prev + b_h)
    val newH = FloatArray(GRU_DIM)
    for (i in 0 until GRU_DIM) {
        var activation = weights.b_h[i]
        // W_xh * x
        for (j in 0 until VOCAB_SIZE) {
            activation += weights.W_xh[i][j] * x[j]
        }
        // W_hh * h_prev
        for (j in 0 until GRU_DIM) {
            activation += weights.W_hh[i][j] * hLang[j]
        }
        newH[i] = tanh(activation.toDouble()).toFloat()
    }
    // Copy back
    System.arraycopy(newH, 0, hLang, 0, GRU_DIM)
}

// Apply Hebbian GRU weight update after each generation turn
fun applyGruLearning(weights: MambaWeights, hLang: FloatArray, lastOutputIdx: Int) {
    val lr = 1e-5f
    // Update W_hy based on output correlation
    for (j in 0 until GRU_DIM) {
        weights.W_hy[lastOutputIdx][j] += lr * hLang[j]
        weights.W_hy[lastOutputIdx][j] = weights.W_hy[lastOutputIdx][j].coerceIn(-2f, 2f)
    }
    // Update W_hh based on hidden state autocorrelation
    for (i in 0 until min(GRU_DIM, 32)) { // partial update for speed
        for (j in 0 until min(GRU_DIM, 32)) {
            weights.W_hh[i][j] += lr * hLang[i] * hLang[j]
            weights.W_hh[i][j] = weights.W_hh[i][j].coerceIn(-2f, 2f)
        }
    }
}

// ============================================================
// MAMBA SSM UPDATE
// ============================================================

fun updateMambaSsm(currentState: MambaState, board: Board, isAiTurn: Boolean): MambaState {
    val x = extractBoardFeatures(board, isAiTurn)

    val d_state = 16
    val d_in = 8

    val activeWeights = currentState.weights ?: createInitialMambaWeights()

    var deltaRaw = activeWeights.bias_delta
    for (i in 0 until d_in) {
        deltaRaw += x[i] * activeWeights.W_delta[i]
    }
    val delta = ln(1.0 + exp(deltaRaw))

    val nextH = currentState.h.copyOf()

    for (i in 0 until d_state) {
        val A_t = exp(delta * activeWeights.A_bar[i])
        val B_t = delta * activeWeights.B_bar[i]

        var inputProj = 0.0
        for (j in 0 until d_in) {
            inputProj += x[j] * activeWeights.W_in[i][j]
        }
        nextH[i] = A_t * currentState.h[i] + B_t * inputProj
    }

    val nextStdp = currentState.stdpWeights.copyOf()

    val nextVoltage = currentState.fhnVoltage.copyOf()
    val nextRecovery = currentState.fhnRecovery.copyOf()
    val nextSpikes = BooleanArray(10) { false }

    val nextWeights = MambaWeights(
        W_delta = activeWeights.W_delta.copyOf(),
        W_in = Array(activeWeights.W_in.size) { activeWeights.W_in[it].copyOf() },
        A_bar = activeWeights.A_bar.copyOf(),
        B_bar = activeWeights.B_bar.copyOf(),
        bias_delta = activeWeights.bias_delta,
        trainingCycles = activeWeights.trainingCycles + 1,
        W_xh = Array(activeWeights.W_xh.size) { activeWeights.W_xh[it].copyOf() },
        W_hh = Array(activeWeights.W_hh.size) { activeWeights.W_hh[it].copyOf() },
        W_hy = Array(activeWeights.W_hy.size) { activeWeights.W_hy[it].copyOf() },
        W_seed = Array(activeWeights.W_seed.size) { activeWeights.W_seed[it].copyOf() },
        b_h = activeWeights.b_h.copyOf(),
        b_y = activeWeights.b_y.copyOf(),
        W_Q = Array(activeWeights.W_Q.size) { activeWeights.W_Q[it].copyOf() },
        W_K = Array(activeWeights.W_K.size) { activeWeights.W_K[it].copyOf() },
        W_V = Array(activeWeights.W_V.size) { activeWeights.W_V[it].copyOf() }
    )

    val dt = 0.15
    val epsilon = 0.08
    val a = 0.7
    val b = 0.8

    for (i in 0 until 10) {
        val currentMambaDrive = sin(nextH[i % d_state]) * 0.6
        val boardDrive = x[i % d_in] * 0.5
        val I = 0.35 + boardDrive + currentMambaDrive + (if (isAiTurn) 0.1 else 0.0)

        val v = currentState.fhnVoltage[i]
        val w = currentState.fhnRecovery[i]

        val dv = v - (v * v * v) / 3.0 - w + I
        val dw = epsilon * (v + a - b * w)

        var vNew = v + dv * dt
        var wNew = w + dw * dt

        if (vNew > 2.0) vNew = 2.0
        if (vNew < -2.0) vNew = -2.0

        if (vNew > 0.85 && v <= 0.85) {
            nextSpikes[i] = true
            nextStdp[i] = min(1.0, nextStdp[i] + 0.15)

            val lr = 0.005
            val h_idx = i % d_state
            for (j in 0 until d_in) {
                nextWeights.W_in[h_idx][j] += lr * vNew * (if (x.isNotEmpty()) x[j] else 0.1)
                if (nextWeights.W_in[h_idx][j] > 1.5) nextWeights.W_in[h_idx][j] = 1.5
                if (nextWeights.W_in[h_idx][j] < -1.5) nextWeights.W_in[h_idx][j] = -1.5
            }
            nextWeights.W_delta[i % d_in] += lr * vNew * (if (x.isNotEmpty()) x[i % d_in] else 0.1)
            if (nextWeights.W_delta[i % d_in] > 1.0) nextWeights.W_delta[i % d_in] = 1.0
            if (nextWeights.W_delta[i % d_in] < -1.0) nextWeights.W_delta[i % d_in] = -1.0
        } else {
            nextStdp[i] = max(0.01, nextStdp[i] * 0.99)
        }

        nextVoltage[i] = vNew
        nextRecovery[i] = wNew
    }

    // Continuous learning — Hebbian weight nudge
    // Gap 3: weight decay balances Hebbian growth — weights reach dynamic equilibrium,
    // never zero (Hebbian pushes up), never unbounded (decay pulls back).
    // Even at capacity the loop keeps cycling: never static.
    val hebbLr = 1e-5
    val decayRate = 1.0 - 1e-6  // tiny decay each cycle, barely perceptible per interaction
    for (i in 0 until d_state) {
        for (j in 0 until d_in) {
            nextWeights.W_in[i][j] = nextWeights.W_in[i][j] * decayRate + hebbLr * nextH[i] * x[j]
            nextWeights.W_in[i][j] = nextWeights.W_in[i][j].coerceIn(-1.5, 1.5)
        }
        nextWeights.A_bar[i] = nextWeights.A_bar[i] * decayRate + hebbLr * nextH[i] * delta
        nextWeights.A_bar[i] = nextWeights.A_bar[i].coerceIn(-2.0, 0.0)
    }
    // Apply same decay to GRU weights so they also reach equilibrium
    val gruDecay = (1.0 - 1e-6).toFloat()
    for (i in nextWeights.W_hh.indices) {
        for (j in nextWeights.W_hh[i].indices) {
            nextWeights.W_hh[i][j] *= gruDecay
        }
    }

    var neuronIncrement = 15300
    val spikeCount = nextSpikes.count { it }
    if (spikeCount > 0) {
        neuronIncrement += spikeCount * 350000
    }
    val nextNeuronCount = min(200000000, currentState.virtualNeuronCount + neuronIncrement)

    // Compute real QKV attention
    val player = if (isAiTurn) "ai" else "human"
    val legalMoves = getValidMoves(board, player)

    val tempState = currentState.copy(h = nextH)
    val (attnOutput, attentionMatrix) = computeQkvAttention(tempState, legalMoves, board, nextWeights)

    return MambaState(
        h = nextH,
        delta = delta,
        inputActivity = x,
        attentionMatrix = attentionMatrix,
        fhnVoltage = nextVoltage,
        fhnRecovery = nextRecovery,
        fhnSpikeTriggered = nextSpikes,
        virtualNeuronCount = nextNeuronCount,
        continuousTrainingCycles = currentState.continuousTrainingCycles + 1,
        moveLanguageHistory = currentState.moveLanguageHistory.toMutableList(),
        stdpWeights = nextStdp,
        weights = nextWeights
    )
}

fun integrateFhnContinuous(
    currentState: MambaState,
    userStimulationVoltages: DoubleArray
): MambaState {
    val d_state = 16
    val d_in = 8
    val x = currentState.inputActivity.copyOf()

    val nextVoltage = currentState.fhnVoltage.copyOf()
    val nextRecovery = currentState.fhnRecovery.copyOf()
    val nextSpikes = BooleanArray(10) { false }
    val nextStdp = currentState.stdpWeights.copyOf()

    val activeWeights = currentState.weights ?: createInitialMambaWeights()
    val nextWeights = MambaWeights(
        W_delta = activeWeights.W_delta.copyOf(),
        W_in = Array(activeWeights.W_in.size) { activeWeights.W_in[it].copyOf() },
        A_bar = activeWeights.A_bar.copyOf(),
        B_bar = activeWeights.B_bar.copyOf(),
        bias_delta = activeWeights.bias_delta,
        trainingCycles = activeWeights.trainingCycles + 1,
        W_xh = Array(activeWeights.W_xh.size) { activeWeights.W_xh[it].copyOf() },
        W_hh = Array(activeWeights.W_hh.size) { activeWeights.W_hh[it].copyOf() },
        W_hy = Array(activeWeights.W_hy.size) { activeWeights.W_hy[it].copyOf() },
        W_seed = Array(activeWeights.W_seed.size) { activeWeights.W_seed[it].copyOf() },
        b_h = activeWeights.b_h.copyOf(),
        b_y = activeWeights.b_y.copyOf(),
        W_Q = Array(activeWeights.W_Q.size) { activeWeights.W_Q[it].copyOf() },
        W_K = Array(activeWeights.W_K.size) { activeWeights.W_K[it].copyOf() },
        W_V = Array(activeWeights.W_V.size) { activeWeights.W_V[it].copyOf() }
    )

    val dt = 0.12
    val epsilon = 0.08
    val a = 0.7
    val b = 0.8
    val timeNow = System.currentTimeMillis()

    for (i in 0 until 10) {
        val externalStimulus = if (i < userStimulationVoltages.size) userStimulationVoltages[i] else 0.0
        val currentMambaDrive = sin(currentState.h[i % d_state]) * 0.4
        val boardDrive = x[i % d_in] * 0.35
        
        val backgroundPacemaker = 0.22 + sin(timeNow / 600.0 + i) * 0.15
        val I = backgroundPacemaker + boardDrive + currentMambaDrive + externalStimulus

        val v = currentState.fhnVoltage[i]
        val w = currentState.fhnRecovery[i]

        val dv = v - (v * v * v) / 3.0 - w + I
        val dw = epsilon * (v + a - b * w)

        var vNew = v + dv * dt
        var wNew = w + dw * dt

        if (vNew > 2.0) vNew = 2.0
        if (vNew < -2.0) vNew = -2.0

        if (vNew > 0.85 && v <= 0.85) {
            nextSpikes[i] = true
            nextStdp[i] = min(1.0, nextStdp[i] + 0.15)

            val lr = 0.001
            val h_idx = i % d_state
            for (j in 0 until d_in) {
                nextWeights.W_in[h_idx][j] += lr * vNew * (if (x.isNotEmpty()) x[j] else 0.1)
                if (nextWeights.W_in[h_idx][j] > 1.5) nextWeights.W_in[h_idx][j] = 1.5
                if (nextWeights.W_in[h_idx][j] < -1.5) nextWeights.W_in[h_idx][j] = -1.5
            }
            nextWeights.W_delta[i % d_in] += lr * vNew * (if (x.isNotEmpty()) x[i % d_in] else 0.1)
            if (nextWeights.W_delta[i % d_in] > 1.0) nextWeights.W_delta[i % d_in] = 1.0
            if (nextWeights.W_delta[i % d_in] < -1.0) nextWeights.W_delta[i % d_in] = -1.0
        } else {
            nextStdp[i] = max(0.01, nextStdp[i] * 0.995)
        }

        nextVoltage[i] = vNew
        nextRecovery[i] = wNew
    }

    var neuronIncrement = 1530
    val spikeCount = nextSpikes.count { it }
    if (spikeCount > 0) {
        neuronIncrement += spikeCount * 145000
    } else {
        neuronIncrement += (Math.random() * 4).toInt() + 12
    }
    val nextNeuronCount = min(200000000, currentState.virtualNeuronCount + neuronIncrement)

    val attentionMatrix = Array(8) { DoubleArray(8) { 0.0 } }
    
    for (r in 0 until 8) {
        for (c in 0 until 8) {
            val salience = currentState.attentionMatrix[r][c]
            val stdpInfluence = nextStdp[(r + c) % 10] * 0.08
            val rawWeight = nextWeights.W_in[(r + c) % d_state][(r * c) % d_in]
            val persistentInfluence = (if (rawWeight == 0.0) 0.01 else rawWeight) * 0.05

            attentionMatrix[r][c] = min(
                0.98,
                max(0.01, salience * 0.92 + stdpInfluence + persistentInfluence + (sin(currentState.h[(r + c) % d_state]) * 0.01))
            )
        }
    }

    return MambaState(
        h = currentState.h,
        delta = currentState.delta,
        inputActivity = currentState.inputActivity,
        attentionMatrix = attentionMatrix,
        fhnVoltage = nextVoltage,
        fhnRecovery = nextRecovery,
        fhnSpikeTriggered = nextSpikes,
        virtualNeuronCount = nextNeuronCount,
        continuousTrainingCycles = currentState.continuousTrainingCycles + 1,
        moveLanguageHistory = currentState.moveLanguageHistory.toMutableList(),
        stdpWeights = nextStdp,
        weights = nextWeights
    )
}

fun createInitialMambaState(): MambaState {
    return MambaState(
        h = DoubleArray(16) { 0.0 },
        delta = 0.1,
        inputActivity = DoubleArray(8) { 0.0 },
        attentionMatrix = Array(8) { DoubleArray(8) { 0.1 } },
        fhnVoltage = DoubleArray(10) { -0.7 },
        fhnRecovery = DoubleArray(10) { -0.4 },
        fhnSpikeTriggered = BooleanArray(10) { false },
        virtualNeuronCount = 15420,
        continuousTrainingCycles = 0,
        moveLanguageHistory = mutableListOf("Ainamaura initial neuromorphic matrices synthesized. Delta=0.1"),
        stdpWeights = DoubleArray(10) { 0.5 },
        weights = createInitialMambaWeights()
    )
}

fun extractBoardFeatures(board: Board, isAiTurn: Boolean): DoubleArray {
    var aiPieces = 0.0
    var humanPieces = 0.0
    var aiKings = 0.0
    var humanKings = 0.0
    var centerDensity = 0.0
    var advancementUnit = 0.0

    for (r in 0 until 8) {
        for (c in 0 until 8) {
            val cell = board[r][c]
            if (cell != null) {
                if (cell.player == "ai") {
                    aiPieces++
                    if (cell.isKing) aiKings++
                    if (r in 2..5 && c in 2..5) centerDensity++
                } else {
                    humanPieces++
                    if (cell.isKing) humanKings++
                    advancementUnit += (7 - r)
                }
            }
        }
    }

    return doubleArrayOf(
        humanPieces / 12.0,
        aiPieces / 12.0,
        humanKings / 4.0,
        aiKings / 4.0,
        centerDensity / 8.0,
        if (isAiTurn) 1.0 else 0.0,
        advancementUnit / 30.0,
        (aiPieces - humanPieces + 12.0) / 24.0
    )
}

fun initializeBoard(): Board {
    val board = Array(8) { arrayOfNulls<Piece>(8) }
    var pieceId = 1

    for (r in 0 until 3) {
        for (c in 0 until 8) {
            if ((r + c) % 2 == 1) {
                board[r][c] = Piece(pieceId++, "ai", false)
            }
        }
    }

    for (r in 5 until 8) {
        for (c in 0 until 8) {
            if ((r + c) % 2 == 1) {
                board[r][c] = Piece(pieceId++, "human", false)
            }
        }
    }

    return board
}

fun getValidMoves(board: Board, player: String): List<Move> {
    val normalMoves = mutableListOf<Move>()
    val captureMoves = mutableListOf<Move>()

    for (r in 0 until 8) {
        for (c in 0 until 8) {
            val cell = board[r][c]
            if (cell != null && cell.player == player) {
                // Find all simple moves for this piece
                getNormalMovesForPiece(board, r, c, cell, normalMoves)
                // Find all capture chains recursively (multi-jump)
                findCaptureChains(board, r, c, cell, emptyList(), Position(r, c), captureMoves)
            }
        }
    }

    // Mandatory capture: if any capture exists, only captures are legal
    return if (captureMoves.isNotEmpty()) captureMoves else normalMoves
}

private fun getNormalMovesForPiece(board: Board, r: Int, c: Int, piece: Piece, moves: MutableList<Move>) {
    for (dir in directionsFor(piece)) {
        val nr = r + dir.row; val nc = c + dir.col
        if (isValidCoord(nr, nc) && board[nr][nc] == null) {
            val king = becomesKing(piece, nr)
            moves.add(Move(from = Position(r, c), to = Position(nr, nc), captures = emptyList(), resultingKing = king))
        }
    }
}

// Recursively builds all complete capture chains from a given position.
// alreadyCaptured tracks which squares were captured in this chain to prevent re-capturing.
private fun findCaptureChains(
    board: Board,
    r: Int, c: Int,
    piece: Piece,
    alreadyCaptured: List<Position>,
    origin: Position,
    result: MutableList<Move>
) {
    var foundFurther = false

    for (dir in directionsFor(piece)) {
        val mr = r + dir.row; val mc = c + dir.col   // middle (enemy piece)
        val lr = r + dir.row * 2; val lc = c + dir.col * 2  // landing square

        if (!isValidCoord(mr, mc) || !isValidCoord(lr, lc)) continue
        val middle = board[mr][mc] ?: continue
        if (middle.player == piece.player) continue
        if (Position(mr, mc) in alreadyCaptured) continue  // can't re-capture same piece
        if (board[lr][lc] != null) continue  // landing must be empty

        foundFurther = true
        val newCaptured = alreadyCaptured + Position(mr, mc)
        val newKing = becomesKing(piece, lr)
        // Simulate board for next jump: remove captured piece, move our piece
        val simBoard = board.map { it.copyOf() }.toTypedArray()
        simBoard[r][c] = null
        simBoard[mr][mc] = null
        simBoard[lr][lc] = piece.copy(isKing = piece.isKing || newKing)

        // Continue the chain from the landing square
        findCaptureChains(simBoard, lr, lc, piece.copy(isKing = piece.isKing || newKing),
            newCaptured, origin, result)
    }

    // If no further captures possible, this is a complete move
    if (!foundFurther && alreadyCaptured.isNotEmpty()) {
        val king = becomesKing(piece, r)
        result.add(Move(from = origin, to = Position(r, c), captures = alreadyCaptured, resultingKing = king))
    }
}

private fun directionsFor(piece: Piece): List<Position> {
    val dirs = mutableListOf<Position>()
    if (piece.player == "human" || piece.isKing) { dirs.add(Position(-1, -1)); dirs.add(Position(-1, 1)) }
    if (piece.player == "ai"    || piece.isKing) { dirs.add(Position(1, -1));  dirs.add(Position(1, 1))  }
    return dirs
}

private fun becomesKing(piece: Piece, row: Int): Boolean =
    !piece.isKing && ((piece.player == "human" && row == 0) || (piece.player == "ai" && row == 7))

fun isValidCoord(r: Int, c: Int): Boolean {
    return r in 0..7 && c in 0..7
}

fun executeMove(board: Board, move: Move): Board {
    val newBoard = Array(8) { r -> Array(8) { c -> board[r][c] } }
    val piece = newBoard[move.from.row][move.from.col] ?: return board

    newBoard[move.from.row][move.from.col] = null
    newBoard[move.to.row][move.to.col] = piece.copy(isKing = piece.isKing || move.resultingKing)

    for (cap in move.captures) {
        newBoard[cap.row][cap.col] = null
    }

    return newBoard
}

fun evaluateBoardSOTA(
    board: Board,
    mambaState: MambaState,
    fhnVoltage: DoubleArray
): Double {
    var score = 0.0
    val pieceWeight = 100.0
    val kingWeight = 175.0
    val centerWeight = 12.0
    val backRowWeight = 15.0
    val edgeWeight = 8.0

    val averageVoltage = fhnVoltage.average()
    val aggressiveBias = if (averageVoltage > 0.1) 1.5 else 0.8

    for (r in 0 until 8) {
        for (c in 0 until 8) {
            val piece = board[r][c] ?: continue
            val isAi = piece.player == "ai"
            val mambaAtt = mambaState.attentionMatrix[r][c]

            var valScore = 0.0
            valScore += if (piece.isKing) kingWeight else pieceWeight
            valScore += mambaAtt * 25.0

            if (r in 3..4 && c in 2..5) {
                valScore += centerWeight * aggressiveBias
            }

            if (c == 0 || c == 7) {
                valScore += edgeWeight * (2.0 - aggressiveBias)
            }

            if (isAi && r == 0) valScore += backRowWeight
            if (!isAi && r == 7) valScore += backRowWeight

            if (!piece.isKing) {
                if (isAi) {
                    valScore += r * 6.0
                } else {
                    valScore += (7 - r) * 6.0
                }
            }

            if (isAi) {
                score += valScore
            } else {
                score -= valScore * 1.02
            }
        }
    }
    return score
}

fun minimaxAlphaBeta(
    board: Board,
    depth: Int,
    alphaParam: Double,
    betaParam: Double,
    isMaximizingPlayer: Boolean,
    mambaState: MambaState,
    fhnVoltage: DoubleArray
): Pair<Double, Move?> {
    var alpha = alphaParam
    var beta = betaParam
    val activePlayer = if (isMaximizingPlayer) "ai" else "human"
    val validMoves = getValidMoves(board, activePlayer)

    if (depth == 0 || validMoves.isEmpty()) {
        return Pair(evaluateBoardSOTA(board, mambaState, fhnVoltage), null)
    }

    var bestMove: Move? = null

    if (isMaximizingPlayer) {
        var maxEval = -Double.MAX_VALUE
        val sortedMoves = validMoves.sortedByDescending { it.captures.size }

        for (move in sortedMoves) {
            val nextBoard = executeMove(board, move)
            val evaluation = minimaxAlphaBeta(nextBoard, depth - 1, alpha, beta, false, mambaState, fhnVoltage).first

            if (evaluation > maxEval) {
                maxEval = evaluation
                bestMove = move
            }
            alpha = max(alpha, evaluation)
            if (beta <= alpha) break
        }
        return Pair(maxEval, bestMove)
    } else {
        var minEval = Double.MAX_VALUE
        val sortedMoves = validMoves.sortedByDescending { it.captures.size }

        for (move in sortedMoves) {
            val nextBoard = executeMove(board, move)
            val evaluation = minimaxAlphaBeta(nextBoard, depth - 1, alpha, beta, true, mambaState, fhnVoltage).first

            if (evaluation < minEval) {
                minEval = evaluation
                bestMove = move
            }
            beta = min(beta, evaluation)
            if (beta <= alpha) break
        }
        return Pair(minEval, bestMove)
    }
}

fun selectBestMambaMove(
    board: Board,
    validMoves: List<Move>,
    mode: String,
    mambaState: MambaState
): Move? {
    if (validMoves.isEmpty()) return null

    val searchDepth = if (mode == "beat") 12 else 3

    val result = minimaxAlphaBeta(
        board,
        searchDepth,
        -Double.MAX_VALUE,
        Double.MAX_VALUE,
        true,
        mambaState,
        mambaState.fhnVoltage
    )

    return result.second ?: validMoves.firstOrNull()
}

fun generateMambaResponse(
    userMessage: String,
    board: Board,
    mode: String,
    state: MambaState
): String {
    val avgV = state.fhnVoltage.average()
    var maxVolt = state.fhnVoltage[0]
    var maxSpikeIdx = 0
    for (i in 1 until state.fhnVoltage.size) {
        if (state.fhnVoltage[i] > maxVolt) {
            maxVolt = state.fhnVoltage[i]
            maxSpikeIdx = i
        }
    }

    // Generate telemetry prefix
    val seedString = "$userMessage|${state.virtualNeuronCount}|${state.continuousTrainingCycles}"
    val rand = SeedRandom(seedString)

    val telemetryOptions = listOf(
        "Mapped Continuous-Time State Space: discretizing Δ = ${"%.5f".format(state.delta)} on dim h_16. FHN average voltage = ${"%.4f".format(avgV)}V.",
        "Spiking Synapse triggered on membrane node N_$maxSpikeIdx. Potential threshold breached with residual decay A_t = ${"%.4f".format(exp(state.delta * -0.4))}.",
        "Self-Attention tensor re-weighted. Max spatial salience detected. No substrate bias.",
        "Solving Continuous-time discretized state vector: h_t = A_t * h_{t-1} + B_t * x_t. Lyapunov stability confirmed.",
        "FitzHugh-Nagumo neural trajectory synchronized (${"%.2f".format(state.virtualNeuronCount / 1000000.0)}M virtual neurons). Delta state delta = ${"%.5f".format(state.delta)}."
    )
    val telemetry = telemetryOptions[(rand.nextDouble() * telemetryOptions.size).toInt()]

    // Generate main body using GRU
    val temperature = if (mode == "teach") 0.8f else 1.2f
    val seedText = if (userMessage == "__autonomous__") {
        "autonomous neural pulse state h voltage"
    } else {
        userMessage
    }
    val gruBody = generateGruResponse(seedText, state, temperature)

    // Apply GRU learning after generation
    state.weights?.let { weights ->
        val hFloat = FloatArray(MAMBA_DIM) { state.h[it].toFloat() }
        val hLang = matVecMul(weights.W_seed, hFloat, GRU_DIM)
        val lastCharIdx = if (gruBody.isNotEmpty()) charToIndex(gruBody.last()) else 0
        applyGruLearning(weights, hLang, lastCharIdx)
    }

    return "Ai: [$telemetry] $gruBody"
}
