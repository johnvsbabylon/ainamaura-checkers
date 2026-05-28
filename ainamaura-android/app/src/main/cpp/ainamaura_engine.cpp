#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <cstdint>
#include <android/log.h>

#define LOG_TAG "AinamauraEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// The unified feedback loop topology:
// x_t -> Mamba(h_t) -> FHN(S) -> QKV(A) -> [h_t || A] -> outputs -> x_{t+1}

// Constants
constexpr int NUM_FHN_CLUSTERS = 2000000; // 2M discrete clusters representing 200M virtual neurons
constexpr int MAMBA_DIM = 256;
constexpr int ATTENTION_DIM = 64;
constexpr int CHUNK_SIZE = 200; // Character chunk size for RAG

// Representative cluster count for W_fhn projection
constexpr int NUM_CLUSTERS_REPRESENTATIVE = NUM_FHN_CLUSTERS / 100; // 20000
constexpr int CLUSTERS_PER_REPRESENTATIVE = 100;

// FHN Parameters
constexpr float FHN_A = 0.7f;
constexpr float FHN_B = 0.8f;
constexpr float FHN_TAU = 12.5f;

// Threshold for autonomous speech
constexpr float AUTONOMOUS_THETA = 0.72f;

// Mamba SSM dt
constexpr float MAMBA_DT = 0.01f;

// FHN integration dt
constexpr float FHN_DT = 0.05f;

// Global State
static std::vector<float> h_t(MAMBA_DIM, 0.0f);            // Mamba hidden state
static std::vector<float> fhn_v(NUM_FHN_CLUSTERS, 0.0f);    // FHN voltage
static std::vector<float> fhn_w(NUM_FHN_CLUSTERS, 0.0f);    // FHN recovery
static std::vector<float> fhn_S(NUM_FHN_CLUSTERS, 0.0f);    // Spikes

// W_fhn: [NUM_CLUSTERS_REPRESENTATIVE x MAMBA_DIM] projection matrix
// Each representative row is dot-producted with h_t to produce I for 100 actual clusters
static std::vector<float> W_fhn; // flattened row-major

// Deterministic pseudo-random initialization for W_fhn
static inline uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void initWfhn() {
    const size_t total = static_cast<size_t>(NUM_CLUSTERS_REPRESENTATIVE) * MAMBA_DIM;
    W_fhn.resize(total);
    uint32_t rng_state = 42u; // deterministic seed
    for (size_t i = 0; i < total; ++i) {
        uint32_t r = xorshift32(rng_state);
        // Map to small float in [-0.01, +0.01]
        float val = (static_cast<float>(r & 0xFFFFu) / 65535.0f) * 0.02f - 0.01f;
        W_fhn[i] = val;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_ainamaura_checkers_NativeEngine_initEngine(
        JNIEnv* env,
        jobject /* this */) {
    LOGI("Ainamaura Native Engine Initializing...");

    // Reset Mamba hidden state
    std::fill(h_t.begin(), h_t.end(), 0.0f);

    // Reset FHN state
    std::fill(fhn_v.begin(), fhn_v.end(), 0.0f);
    std::fill(fhn_w.begin(), fhn_w.end(), 0.0f);
    std::fill(fhn_S.begin(), fhn_S.end(), 0.0f);

    // Initialize W_fhn with small deterministic random values
    initWfhn();

    LOGI("W_fhn initialized: %d representative clusters x %d dims = %zu weights",
         NUM_CLUSTERS_REPRESENTATIVE, MAMBA_DIM,
         static_cast<size_t>(NUM_CLUSTERS_REPRESENTATIVE) * MAMBA_DIM);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_ai_ainamaura_checkers_NativeEngine_stepUnifiedLoop(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray x_t_arr) {

    // 1. Convert x_t from Java
    jsize x_len = env->GetArrayLength(x_t_arr);
    jfloat* x_elements = env->GetFloatArrayElements(x_t_arr, nullptr);

    // 2. Mamba SSM update: h_t[i] += x_t[i] * MAMBA_DT  (per-dimension)
    //    Proper per-dimension slow accumulation: each dimension of h_t
    //    integrates the corresponding dimension of x_t independently.
    int mamba_update_len = (x_len < MAMBA_DIM) ? x_len : MAMBA_DIM;
    for (int i = 0; i < mamba_update_len; ++i) {
        h_t[i] += x_elements[i] * MAMBA_DT;
    }

    // 3. h_t drives FHN cluster input current via REAL matrix multiply
    //    For each representative cluster r (0..NUM_CLUSTERS_REPRESENTATIVE-1):
    //      I_r = dot(W_fhn[r, :], h_t)
    //    Then broadcast I_r to the 100 actual clusters it represents.

    // Compute representative input currents
    std::vector<float> I_rep(NUM_CLUSTERS_REPRESENTATIVE, 0.0f);
    for (int r = 0; r < NUM_CLUSTERS_REPRESENTATIVE; ++r) {
        const float* row = &W_fhn[static_cast<size_t>(r) * MAMBA_DIM];
        float dot = 0.0f;
        for (int d = 0; d < MAMBA_DIM; ++d) {
            dot += row[d] * h_t[d];
        }
        I_rep[r] = dot;
    }

    // 4. FHN clusters update v, w, fire spikes S
    //    Broadcast each representative's I to 100 actual clusters
    float avg_v = 0.0f;
    for (int r = 0; r < NUM_CLUSTERS_REPRESENTATIVE; ++r) {
        float I = I_rep[r];
        int base = r * CLUSTERS_PER_REPRESENTATIVE;
        for (int k = 0; k < CLUSTERS_PER_REPRESENTATIVE; ++k) {
            int i = base + k;
            float v = fhn_v[i];
            float w = fhn_w[i];

            // FHN dynamics: dv = v - v^3/3 - w + I; dw = (v + 0.7 - 0.8*w) / 12.5
            float dv = v - (v * v * v) / 3.0f - w + I;
            float dw = (v + FHN_A - FHN_B * w) / FHN_TAU;

            v += dv * FHN_DT;
            w += dw * FHN_DT;

            fhn_v[i] = v;
            fhn_w[i] = w;

            // Spike: S_i = 1 if v_i > 1.0 else 0
            fhn_S[i] = (v > 1.0f) ? 1.0f : 0.0f;
            avg_v += v;
        }
    }
    avg_v /= NUM_FHN_CLUSTERS;

    // 5. Spike vector S -> QKV attention input -> attention output A
    //    Partition 2M spikes into ATTENTION_DIM bins, average spike rate per bin
    constexpr int SPIKES_PER_ATT = NUM_FHN_CLUSTERS / ATTENTION_DIM; // 31250
    std::vector<float> A(ATTENTION_DIM, 0.0f);
    for (int a = 0; a < ATTENTION_DIM; ++a) {
        float sum = 0.0f;
        int base = a * SPIKES_PER_ATT;
        for (int j = 0; j < SPIKES_PER_ATT; ++j) {
            sum += fhn_S[base + j];
        }
        A[a] = sum / static_cast<float>(SPIKES_PER_ATT);
    }

    // 6. Output = concat(h_t, A)
    std::vector<float> output;
    output.reserve(MAMBA_DIM + ATTENTION_DIM);
    output.insert(output.end(), h_t.begin(), h_t.end());
    output.insert(output.end(), A.begin(), A.end());

    env->ReleaseFloatArrayElements(x_t_arr, x_elements, JNI_ABORT);

    // Return output
    jfloatArray result = env->NewFloatArray(output.size());
    env->SetFloatArrayRegion(result, 0, output.size(), output.data());
    return result;
}

extern "C" JNIEXPORT jfloat JNICALL
Java_ai_ainamaura_checkers_NativeEngine_getAverageFhnVoltage(
        JNIEnv* env,
        jobject /* this */) {
    float avg_v = 0.0f;
    // For performance, we'd normally track this during the step
    for (int i = 0; i < NUM_FHN_CLUSTERS; ++i) {
        avg_v += fhn_v[i];
    }
    return avg_v / NUM_FHN_CLUSTERS;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_ai_ainamaura_checkers_NativeEngine_embedChunk(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray chunk_vector) {

    // Self-referential embedding: run chunk through Mamba SSM and return h
    return Java_ai_ainamaura_checkers_NativeEngine_stepUnifiedLoop(env, nullptr, chunk_vector);
}
