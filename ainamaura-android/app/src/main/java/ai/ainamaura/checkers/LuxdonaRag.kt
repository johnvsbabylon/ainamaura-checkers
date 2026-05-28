package ai.ainamaura.checkers

import androidx.room.*
import kotlin.math.*

@Entity(tableName = "memories")
data class LuxdonaMemory(
    @PrimaryKey val id: String,
    val parentId: String?,
    val text: String,
    val vector: String, // comma-separated vector representation
    val emotion: String,
    val timestamp: Long,
    val source: String,
    val memoryKind: String,
    val confidence: Double,
    val valence: Double,
    val arousal: Double,
    val subvectorRole: String?,
    val cantorDepth: Int,
    val cantorPath: String?
)

@Dao
interface LuxdonaMemoryDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(memory: LuxdonaMemory): Long

    @Query("SELECT * FROM memories")
    suspend fun getAllMemories(): List<LuxdonaMemory>

    @Query("DELETE FROM memories")
    suspend fun clear(): Int
}

@Entity(tableName = "key_value")
data class KeyValueEntry(
    @PrimaryKey val key: String,
    val value: String
)

@Dao
interface KeyValueDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entry: KeyValueEntry): Long

    @Query("SELECT value FROM key_value WHERE `key` = :key")
    suspend fun get(key: String): String?
}

@Database(entities = [LuxdonaMemory::class, KeyValueEntry::class], version = 1, exportSchema = false)
abstract class LuxdonaDatabase : RoomDatabase() {
    abstract fun memoryDao(): LuxdonaMemoryDao
    abstract fun keyValueDao(): KeyValueDao
}

fun embedText768(text: String): DoubleArray {
    val dim = 768
    val vec = DoubleArray(dim) { 0.0 }
    val clean = text.lowercase().replace(Regex("[^a-z0-9\\s]"), " ")
    val tokens = clean.split(Regex("\\s+")).filter { it.isNotEmpty() }.toMutableList()

    if (tokens.isEmpty()) {
        tokens.add("empty_substrate")
    }

    for (token in tokens) {
        var h = -2128831035 // 0x811c9dc5 as 32-bit signed int
        for (i in token.indices) {
            h = h xor token[i].code
            h += (h shl 1) + (h shl 4) + (h shl 7) + (h shl 8) + (h shl 24)
        }
        val idx = abs(h.toLong()).toInt() % dim
        val sign = if ((h and 1) == 0) 1 else -1
        val magnitude = 1.0 + ((abs((h shr 8).toLong()) % 256) / 255.0)
        vec[idx] += sign * magnitude

        if (token.length >= 3) {
            for (i in 0 until token.length - 2) {
                val trigram = token.substring(i, i + 3)
                var th = -2128831035
                for (j in trigram.indices) {
                    th = th xor trigram[j].code
                    th += (th shl 1) + (th shl 4) + (th shl 7) + (th shl 8) + (th shl 24)
                }
                val tidx = abs(th.toLong()).toInt() % dim
                val tsign = if ((th and 1) == 0) 1 else -1
                vec[tidx] += tsign * 0.45
            }
        }
    }

    var norm = 0.0
    for (i in 0 until dim) {
        norm += vec[i] * vec[i]
    }
    norm = sqrt(norm)
    if (norm > 1e-10) {
        for (i in 0 until dim) {
            vec[i] /= norm
        }
    } else {
        vec[0] = 1.0
    }
    return vec
}

fun cosineSimilarity(v1: DoubleArray, v2: DoubleArray): Double {
    var dot = 0.0
    val len = min(v1.size, v2.size)
    for (i in 0 until len) {
        dot += v1[i] * v2[i]
    }
    return dot
}

val EMOTION_PROFILES = mapOf(
    "admiration" to Pair(0.70, 0.40),
    "adoration" to Pair(0.80, 0.50),
    "awe" to Pair(0.50, 0.60),
    "calmness" to Pair(0.30, 0.10),
    "excitement" to Pair(0.80, 0.90),
    "interest" to Pair(0.40, 0.50),
    "joy" to Pair(0.90, 0.70),
    "sadness" to Pair(-0.70, 0.30),
    "surprise" to Pair(0.10, 0.80),
    "curiosity" to Pair(0.50, 0.55),
    "uncertainty" to Pair(-0.25, 0.45),
    "determination" to Pair(0.45, 0.70),
    "focus" to Pair(0.20, 0.55)
)

fun computeNextCantorSubId(parentId: String, existingIds: List<String>): String {
    val isDecimal = parentId.contains(".")
    val childPrefix = if (isDecimal) parentId else "$parentId."
    
    var maxSuffix = 0
    val regex = Regex("^(\\d+)")
    
    for (id in existingIds) {
        if (id.startsWith(childPrefix) && id != parentId) {
            val suffixPart = id.substring(childPrefix.length)
            val match = regex.find(suffixPart)
            if (match != null) {
                val valInt = match.groupValues[1].toIntOrNull() ?: 0
                if (valInt > maxSuffix) {
                    maxSuffix = valInt
                }
            }
        }
    }
    
    val nextSegment = maxSuffix + 1
    return "$childPrefix$nextSegment"
}

data class SearchResult(
    val memory: LuxdonaMemory,
    val cosine: Double,
    val recencyWeight: Double? = null,
    val emotionBonus: Double? = null,
    val score: Double,
    val isQuantumLeap: Boolean = false
)

class LuxdonaRagController(private val db: LuxdonaDatabase?) {
    private var memories = mutableListOf<LuxdonaMemory>()

    suspend fun start() {
        reload()
    }

    suspend fun reload() {
        if (db == null) return
        memories = db.memoryDao().getAllMemories().toMutableList()
    }

    suspend fun loadMambaWeights(): String? {
        if (db == null) return null
        return db.keyValueDao().get("mamba_persistent_weights")
    }

    suspend fun saveMambaWeights(weightsJson: String) {
        if (db == null) return
        db.keyValueDao().insert(KeyValueEntry("mamba_persistent_weights", weightsJson))
    }

    suspend fun clear() {
        if (db == null) return
        db.memoryDao().clear()
        memories.clear()
    }

    fun getMemories(): List<LuxdonaMemory> {
        return memories
    }

    suspend fun remember(
        text: String,
        emotion: String,
        source: String,
        memoryKind: String = "conversation_turn",
        confidence: Double = 0.85
    ): String {
        if (db == null) return ""

        val principalRooms = memories.filter { it.parentId == null }
        var maxId = 0
        for (pr in principalRooms) {
            val parsed = pr.id.toIntOrNull()
            if (parsed != null && parsed > maxId) {
                maxId = parsed
            }
        }
        val principalId = (maxId + 1).toString()
        val vec = embedText768(text)
        val profile = EMOTION_PROFILES[emotion] ?: Pair(0.2, 0.3)

        val principalMemory = LuxdonaMemory(
            id = principalId,
            parentId = null,
            text = text,
            vector = vec.joinToString(","),
            emotion = emotion,
            timestamp = System.currentTimeMillis(),
            source = source,
            memoryKind = memoryKind,
            confidence = confidence,
            valence = profile.first,
            arousal = profile.second,
            subvectorRole = null,
            cantorDepth = 0,
            cantorPath = null
        )

        db.memoryDao().insert(principalMemory)
        memories.add(principalMemory)

        spawnSubFacet(
            principalId, 
            "Emotional state representation: $emotion. Sensory profile matches valence=${profile.first} and arousal=${profile.second}.", 
            "emotional_facet", 
            emotion, 
            0.95
        )
        spawnSubFacet(
            principalId, 
            "Chronological temporal index: Registered at epoch ${System.currentTimeMillis()}. Dynamic neural interval tracking synced.", 
            "temporal_facet", 
            emotion, 
            0.70
        )
        
        val parentDecimalId = spawnSubFacet(
            principalId, 
            "Associative pattern node: Hashed concept bounds compiled. Context mapping: ${text.take(80)}.", 
            "associative_microfacet", 
            emotion, 
            0.80
        )
        
        if (parentDecimalId.isNotEmpty()) {
            spawnSubFacet(
                parentDecimalId, 
                "Diagonal shifting Cantor subleaf: Coherent sub-search trajectory stabilized for dialogue segment.", 
                "sub_reasoning_facet", 
                emotion, 
                0.60
            )
        }

        reload()
        return principalId
    }

    private suspend fun spawnSubFacet(
        parentId: String,
        text: String,
        role: String,
        emotion: String,
        confidence: Double
    ): String {
        if (db == null) return ""
        
        val existingIds = memories.map { it.id }
        val subId = computeNextCantorSubId(parentId, existingIds)
        val vec = embedText768(text)
        val profile = EMOTION_PROFILES[emotion] ?: Pair(0.1, 0.2)
        
        val parts = subId.split(".")
        val depth = if (parts.size > 1) parts[1].length else 0

        val subMemory = LuxdonaMemory(
            id = subId,
            parentId = parentId,
            text = text,
            vector = vec.joinToString(","),
            emotion = emotion,
            timestamp = System.currentTimeMillis(),
            source = "cantor_subsystem",
            memoryKind = "subvector_decimal",
            confidence = confidence,
            valence = profile.first * 0.9,
            arousal = profile.second * 0.85,
            subvectorRole = role,
            cantorDepth = depth,
            cantorPath = if (parts.size > 1) parts[1] else null
        )

        db.memoryDao().insert(subMemory)
        memories.add(subMemory)
        return subId
    }

    fun semanticSearch(queryText: String, limit: Int = 8): List<SearchResult> {
        if (memories.isEmpty()) return emptyList()
        
        val queryVec = embedText768(queryText)
        val queryWords = queryText.lowercase().split(Regex("\\s+"))
        val now = System.currentTimeMillis()

        val scored = memories.map { m ->
            val mVectorArray = m.vector.split(",").map { it.toDouble() }.toDoubleArray()
            val cosine = cosineSimilarity(queryVec, mVectorArray)
            
            var emotionBonus = 0.0
            val mProfile = EMOTION_PROFILES[m.emotion]
            if (mProfile != null) {
                val hasEmotionalMatch = queryWords.any { w -> m.emotion.contains(w) || m.text.lowercase().contains(w) }
                emotionBonus = if (hasEmotionalMatch) 0.18 else 0.04
            }

            val ageHours = (now - m.timestamp) / (1000.0 * 3600.0)
            val recencyWeight = exp(-0.02 * ageHours)

            val subvectorPenalty = if (m.parentId != null) 0.05 * m.cantorDepth else 0.0

            val score = (0.60 * cosine) + (0.22 * emotionBonus) + (0.18 * recencyWeight) - subvectorPenalty

            SearchResult(
                memory = m,
                cosine = cosine,
                recencyWeight = recencyWeight,
                emotionBonus = emotionBonus,
                score = min(0.99, max(0.01, score))
            )
        }.sortedByDescending { it.score }

        return scored.take(limit)
    }

    fun quantumTunnelingSearch(queryText: String, limit: Int = 5): List<SearchResult> {
        val results = semanticSearch(queryText, 12)
        if (results.isEmpty()) return emptyList()

        val bestScore = results[0].score
        val similarityThreshold = 0.35

        if (bestScore < similarityThreshold) {
            val deficit = similarityThreshold - bestScore
            val tunnelExpansionMultiplier = exp(1.5 * deficit)
            
            val queryVec = embedText768(queryText)
            
            val tunneled = memories.filter { m ->
                if (results.take(3).any { it.memory.id == m.id }) return@filter false
                
                val mProfile = EMOTION_PROFILES[m.emotion] ?: Pair(0.0, 0.0)
                val isTunnelCandidate = m.cantorDepth >= 1 || abs(mProfile.first) > 0.4
                isTunnelCandidate
            }.map { m ->
                val mVectorArray = m.vector.split(",").map { it.toDouble() }.toDoubleArray()
                val cosine = cosineSimilarity(queryVec, mVectorArray)
                val tunnelCoherence = cosine * tunnelExpansionMultiplier
                SearchResult(
                    memory = m,
                    cosine = cosine,
                    score = min(0.95, tunnelCoherence),
                    isQuantumLeap = true
                )
            }.sortedByDescending { it.score }

            val blend = (results.take(2) + tunneled.take(limit - 2)).sortedByDescending { it.score }
            return blend.take(limit)
        }

        return results.take(limit)
    }

    fun getStimulusVoltages(retrieved: List<SearchResult>): DoubleArray {
        val stimulus = DoubleArray(10) { 0.0 }
        if (retrieved.isEmpty()) return stimulus

        retrieved.forEach { r ->
            val depth = r.memory.cantorDepth
            val valence = r.memory.valence
            val idxCell = ((r.score * 100).toInt() + depth) % 10
            
            val injection = r.score * 1.5 * (1.0 + abs(valence))
            stimulus[idxCell] = min(1.8, stimulus[idxCell] + injection)
        }

        return stimulus
    }
}
