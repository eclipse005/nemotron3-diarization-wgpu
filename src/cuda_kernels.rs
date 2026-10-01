//! CUDA device code compiled at runtime with NVRTC (`compute_61` / Pascal).

pub const SRC: &str = r#"
extern "C" {

__global__ void layer_norm(const float* x, const float* w, const float* b, float* y,
                           int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* src = x + (size_t)row * cols;
    float* dst = y + (size_t)row * cols;
    float sum = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sum += src[i];
    __shared__ float red[256];
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float mean = red[0] / (float)cols;
    __syncthreads();
    float var = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float d = src[i] - mean;
        var += d * d;
    }
    red[threadIdx.x] = var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)cols + 1e-5f);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        dst[i] = (src[i] - mean) * inv * w[i] + b[i];
    }
}

__global__ void gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.f + erff(v * 0.7071067811865476f));
}

__global__ void relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = fmaxf(x[i], 0.f);
}

__global__ void add_inplace(float* c, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] += b[i];
}

__global__ void bias_add(float* y, const float* bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i >= n) return;
    y[i] += bias[i % cols];
}

__global__ void rope(float* x, const float* cos, const float* sin,
                     int seq, int heads, int hd, int stride, int col_off) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = seq * heads;
    if (idx >= n) return;
    int t = idx / heads;
    int head = idx % heads;
    int half = hd / 2;
    int base = t * stride + col_off + head * hd;
    int cbase = t * hd;
    for (int i = 0; i < half; i++) {
        float x1 = x[base + i];
        float x2 = x[base + half + i];
        x[base + i] = x1 * cos[cbase + i] - x2 * sin[cbase + i];
        x[base + half + i] = x2 * cos[cbase + half + i] + x1 * sin[cbase + half + i];
    }
}

__global__ void add_bias(float* x, const float* y, const float* bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i >= n) return;
    x[i] += y[i] + bias[i % cols];
}

__global__ void bias_gelu(float* x, const float* bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i >= n) return;
    float v = x[i] + bias[i % cols];
    x[i] = 0.5f * v * (1.f + erff(v * 0.7071067811865476f));
}

__global__ void softmax_rows(float* s, int seq, int heads, const int* valid_ptr) {
    int q = blockIdx.x;
    int h = blockIdx.y;
    int valid = *valid_ptr;
    if (q >= seq || h >= heads) return;
    float* row = s + ((size_t)h * seq + q) * seq;
    float m = -1e30f;
    for (int j = threadIdx.x; j < valid; j += blockDim.x) m = fmaxf(m, row[j]);
    __shared__ float red[256];
    red[threadIdx.x] = m;
    __syncthreads();
    for (int t = blockDim.x / 2; t > 0; t >>= 1) {
        if (threadIdx.x < t) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + t]);
        __syncthreads();
    }
    float mx = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int j = threadIdx.x; j < valid; j += blockDim.x) {
        float e = expf(row[j] - mx);
        row[j] = e;
        sum += e;
    }
    for (int j = valid + threadIdx.x; j < seq; j += blockDim.x) row[j] = 0.f;
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int t = blockDim.x / 2; t > 0; t >>= 1) {
        if (threadIdx.x < t) red[threadIdx.x] += red[threadIdx.x + t];
        __syncthreads();
    }
    float inv = 1.f / red[0];
    for (int j = threadIdx.x; j < valid; j += blockDim.x) row[j] *= inv;
}

__global__ void conv1d_k3(const float* x, const float* w, const float* bias, float* y,
                          int frames, int hh, int out_c) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= frames || oc >= out_c) return;
    float acc = bias[oc];
    for (int ic = 0; ic < hh; ic++) {
        int wbase = (oc * hh + ic) * 3;
        float left = t > 0 ? x[(t - 1) * hh + ic] : 0.f;
        float mid = x[t * hh + ic];
        float right = (t + 1 < frames) ? x[(t + 1) * hh + ic] : 0.f;
        acc += w[wbase] * left + w[wbase + 1] * mid + w[wbase + 2] * right;
    }
    y[t * out_c + oc] = acc;
}

}
"#;