#include "winograd.cuh"

// Transformation matrices for F(2x2, 3x3)
__constant__ float G[4][3] = {
    {1.0f, 0.0f, 0.0f}, 
    {0.5f, 0.5f, 0.5f}, 
    {0.5f, -0.5f, 0.5f}, 
    {0.0f, 0.0f, 1.0f}
};

__constant__ float B_T[4][4] = {
    {1.0f, 0.0f, -1.0f, 0.0f}, 
    {0.0f, 1.0f, 1.0f, 0.0f}, 
    {0.0f, -1.0f, 1.0f, 0.0f}, 
    {0.0f, 1.0f, 0.0f, -1.0f}
};

__constant__ float B[4][4] = {
    {1.0f,  0.0f,  0.0f,  0.0f}, 
    {0.0f,  1.0f, -1.0f,  1.0f}, 
    {-1.0f, 1.0f,  1.0f,  0.0f}, 
    {0.0f,  0.0f,  0.0f, -1.0f}
};

__constant__ float A_T[2][4] = {
    {1.0f, 1.0f, 1.0f, 0.0f}, 
    {0.0f, 1.0f, -1.0f, -1.0f}
};

// 优化的输入变换 - 直接计算v = B^T @ d @ B，避免中间矩阵
__device__ __forceinline__ void optimized_input_transform(const float d[16], float v[16]) {
    // 根据测试程序输出的正确系数计算 v = B^T @ d @ B
    v[0] = d[0] - d[2] - d[8] + d[10];
    v[1] = d[1] + d[2] - d[9] - d[10];
    v[2] = -d[1] + d[2] + d[9] - d[10];
    v[3] = d[1] - d[3] - d[9] + d[11];
    
    v[4] = d[4] - d[6] + d[8] - d[10];
    v[5] = d[5] + d[6] + d[9] + d[10];
    v[6] = -d[5] + d[6] - d[9] + d[10];
    v[7] = d[5] - d[7] + d[9] - d[11];
    
    v[8] = -d[4] + d[6] + d[8] - d[10];
    v[9] = -d[5] - d[6] + d[9] + d[10];
    v[10] = d[5] - d[6] - d[9] + d[10];
    v[11] = -d[5] + d[7] + d[9] - d[11];
    
    v[12] = d[4] - d[6] - d[12] + d[14];
    v[13] = d[5] + d[6] - d[13] - d[14];
    v[14] = -d[5] + d[6] + d[13] - d[14];
    v[15] = d[5] - d[7] - d[13] + d[15];
}

// 优化的输出变换 - 直接计算Y = A^T @ m @ A，避免中间矩阵
__device__ __forceinline__ void optimized_output_transform(const float m[16], float Y[4]) {
    // 根据测试程序输出的正确系数计算 Y = A^T @ m @ A
    Y[0] = m[0] + m[1] + m[2] + m[4] + m[5] + m[6] + m[8] + m[9] + m[10];
    Y[1] = m[1] - m[2] - m[3] + m[5] - m[6] - m[7] + m[9] - m[10] - m[11];
    Y[2] = m[4] + m[5] + m[6] - m[8] - m[9] - m[10] - m[12] - m[13] - m[14];
    Y[3] = m[5] - m[6] - m[7] - m[9] + m[10] + m[11] - m[13] + m[14] + m[15];
}

// Kernel to precompute filter transformations
__global__
void filter_transform_kernel(const float* __restrict__ filter,
                             float* __restrict__ U,
                             int K, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_filters = K * C;
    if (idx >= total_filters) return;
    
    int k = idx / C;
    int c = idx % C;
    
    // Get pointer to the 3x3 filter for (k, c)
    const float* g = filter + (k * C + c) * 9;
    
    // Get pointer to output 4x4 transformed filter
    float* u_kc = U + (k * C + c) * 16;
    
    // Filter Transform: U = G * g * G^T
    float temp_g[4][3];
    
    // First step: temp_g = G * g
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            temp_g[i][j] = G[i][0] * g[0 * 3 + j] + G[i][1] * g[1 * 3 + j] + G[i][2] * g[2 * 3 + j];
        }
    }
    
    // Second step: u_kc = temp_g * G^T (manually computed G^T multiplication)
    for (int i = 0; i < 4; ++i) {
        u_kc[i * 4 + 0] = temp_g[i][0];
        u_kc[i * 4 + 1] = 0.5f * (temp_g[i][0] + temp_g[i][1] + temp_g[i][2]);
        u_kc[i * 4 + 2] = 0.5f * (temp_g[i][0] - temp_g[i][1] + temp_g[i][2]);
        u_kc[i * 4 + 3] = temp_g[i][2];
    }
}




// Fused kernel for Winograd convolution F(2x2, 3x3) using precomputed filter transforms with shared memory optimization
__global__
void winograd_conv_kernel(const float* __restrict__ image,
                          const float* __restrict__ filter,
                          float* __restrict__ output,
                          int N, int C, int H, int W, int K, int outH, int outW) {
    // 共享内存声明
    extern __shared__ float shared_memory[];
    
    // 计算共享内存布局
    const int input_tile_h = blockDim.y * 2 + 2;  // 高度方向的输入tile大小
    const int input_tile_w = blockDim.x * 2 + 2;  // 宽度方向的输入tile大小
    const int input_shared_size = input_tile_h * input_tile_w;  // 每个通道的输入数据大小
    const int filter_shared_size = 16 * blockDim.z;  // 变换后的卷积核大小 (4x4 * blockDim.z个输出通道)
    
    // 共享内存指针
    float* shared_input = shared_memory;  // [input_tile_h][input_tile_w]
    float* shared_filters = shared_memory + input_shared_size;  // [blockDim.z][16]
    
    // 负责所有批次，所有通道，在空间和输出维度上并行化
    // thread[k][y][x] -> InputMatrix[:][:][2*y][2*x] 
    // thread[k][y][x] -> Kernel[k][:][y][x]                      
    // for n in batches:
    //      acc = 0
    //      for c in channels:
    //          sync_load InputMatrix[n][c][start_y:end_y][start_x:end_x]  // shared_input
    //          sync_load Kernel[start_k:end_k][c][:][:]    // kernel_size = 16 * blockDim.z
    //          sync_threads
    //          
    //          temp = InputMatrix[n][c][y*2:y*2+4][x*2:x*2+4] 比如x=0时 对应0:4
    //          u = Kernel[k][c][:][:]
    //          v = B^T @ temp @ B
    //          acc += v * u
    //      end for;
    //      output[n][k][y*2:y*2+4][x*2:x*2+4] = A^T @ acc @ A
    // end for;
    // 线程映射: x=tile_x, y=tile_y, z=output_channel
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;  // 输出通道索引
    
    const int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y * blockDim.z;                    
    int tiles_x = (outW + 1) / 2;
    int tiles_y = (outH + 1) / 2;
    
    // 注意：不要在这里提前返回，所有线程都需要参与协作加载

    // 串行处理每个 batch
    for (int n = 0; n < N; ++n) {
        // 使用单个累加器数组
        float accumulator[16] = {0.0f};

                    // 循环处理输入通道
            for (int c = 0; c < C; ++c) {
                __syncthreads();
                
                // --- 优化的串行加载策略（保持内存合并访问）---
                // 先加载卷积核（较小的数据，快速完成）
                for (int load_idx = tid; load_idx < filter_shared_size; load_idx += total_threads) {
                    int k_local = load_idx / 16;
                    int filter_elem = load_idx % 16;
                    int k_global = blockIdx.z * blockDim.z + k_local;
                    
                    if (k_global < K) {
                        const float* u_kc = filter + (k_global * C + c) * 16;
                        shared_filters[load_idx] = u_kc[filter_elem];
                    } else {
                        shared_filters[load_idx] = 0.0f;
                    }
                }
                
                // 然后加载输入数据（保持连续的内存访问模式）
                int input_start_h = blockIdx.y * blockDim.y * 2;
                int input_start_w = blockIdx.x * blockDim.x * 2;
                
                for (int load_idx = tid; load_idx < input_shared_size; load_idx += total_threads) {
                    int local_h = load_idx / input_tile_w;
                    int local_w = load_idx % input_tile_w;
                    int global_h = input_start_h + local_h;
                    int global_w = input_start_w + local_w;
                    
                    if (global_h >= 0 && global_h < H && global_w >= 0 && global_w < W) {
                        shared_input[load_idx] = image[(n * C + c) * H * W + global_h * W + global_w];
                    } else {
                        shared_input[load_idx] = 0.0f;  // Zero padding
                    }
                }
                
                __syncthreads(); // 确保两个加载都完成
            
            // --- 从共享内存进行Winograd变换和计算 ---
            // 计算当前线程对应的输入数据在共享内存中的位置
            int local_h_start = threadIdx.y * 2;
            int local_w_start = threadIdx.x * 2;
            
            // 确保访问不越界，且当前线程在有效范围内
            if (tile_x < tiles_x && tile_y < tiles_y && k < K && 
                local_h_start + 3 < input_tile_h && local_w_start + 3 < input_tile_w) {
                // 提取4x4输入块并直接进行优化的输入变换
                float d[16], v[16];
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        d[i * 4 + j] = shared_input[(local_h_start + i) * input_tile_w + (local_w_start + j)];
                    }
                }
                
                // 使用优化的输入变换，避免中间矩阵
                optimized_input_transform(d, v);
                
                // 获取对应的变换卷积核并进行逐元素乘积累加
                int k_local = threadIdx.z;  // 块内的输出通道索引
                if (k_local < blockDim.z && (blockIdx.z * blockDim.z + k_local) < K) {
                    const float* u_kc = shared_filters + k_local * 16;
                    
                    // acc += v * u (逐元素相乘)
                    for (int i = 0; i < 16; ++i) {
                        accumulator[i] += v[i] * u_kc[i];
                    }
                }
            }
        }

        // --- 使用优化的输出变换 ---（只有有效线程进行输出变换和写入）
        if (tile_x < tiles_x && tile_y < tiles_y && k < K) {
            // 使用优化的输出变换，避免中间矩阵
            float Y[4];
            optimized_output_transform(accumulator, Y);
            
            // 步骤 3: 写入最终输出
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    int h = tile_y * 2 + i;
                    int w = tile_x * 2 + j;
                    if (h < outH && w < outW) {
                        output[((n * K + k) * outH + h) * outW + w] = Y[i * 2 + j];
                    }
                }
            }
        }
    }
}



// 高效内联版本的1D Winograd卷积核函数 - 避免函数调用开销
__global__
void winograd_conv_kernel_1D_inline(const float* __restrict__ image,
                                   const float* __restrict__ filter,
                                   float* __restrict__ output,
                                   int N, int C, int H, int W, int K, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    if (idx >= num_tiles) return;

    // 解析线程索引
    int p_local = idx % ((outH / 2) * (outW / 2));
    int k = (idx / ((outH / 2) * (outW / 2))) % K;
    int n = idx / (K * (outH / 2) * (outW / 2));
    int tile_y = p_local / (outW / 2);
    int tile_x = p_local % (outW / 2);

    // SIMD对齐的局部累加矩阵
    __align__(16) float m_flat[16] = {0.0f};

    // 循环处理所有输入通道
    for (int c = 0; c < C; ++c) {
        // 获取预计算的滤波器变换
        const float* u_kc = filter + (k * C + c) * 16;
        
        // === SIMD优化数据加载：使用float4向量化加载 ===
        __align__(16) float d_flat[16];
        const float* base = image + (n * C + c) * H * W;
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        
        // 安全的数据加载：避免内存对齐问题
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int global_h = h_start + i;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int global_w = w_start + j;
                d_flat[i * 4 + j] = (global_h >= 0 && global_h < H && global_w >= 0 && global_w < W) 
                                   ? base[global_h * W + global_w] : 0.0f;
            }
        }
        
        // === SIMD优化输入变换：使用float4向量化计算 ===
        __align__(16) float v_flat[16];
        
        // 向量化加载d矩阵的行
        float4 d_row0 = *((float4*)&d_flat[0]);   // d[0-3]
        float4 d_row1 = *((float4*)&d_flat[4]);   // d[4-7]  
        float4 d_row2 = *((float4*)&d_flat[8]);   // d[8-11]
        float4 d_row3 = *((float4*)&d_flat[12]);  // d[12-15]
        
        // 向量化计算第一组：v[0-3]
        float4 v_row0 = make_float4(
            d_row0.x - d_row0.z - d_row2.x + d_row2.z,    // v[0] = d[0] - d[2] - d[8] + d[10]
            d_row0.y + d_row0.z - d_row2.y - d_row2.z,    // v[1] = d[1] + d[2] - d[9] - d[10]
            -d_row0.y + d_row0.z + d_row2.y - d_row2.z,   // v[2] = -d[1] + d[2] + d[9] - d[10]
            d_row0.y - d_row0.w - d_row2.y + d_row2.w     // v[3] = d[1] - d[3] - d[9] + d[11]
        );
        
        // 向量化计算第二组：v[4-7]
        float4 v_row1 = make_float4(
            d_row1.x - d_row1.z + d_row2.x - d_row2.z,    // v[4] = d[4] - d[6] + d[8] - d[10]
            d_row1.y + d_row1.z + d_row2.y + d_row2.z,    // v[5] = d[5] + d[6] + d[9] + d[10]
            -d_row1.y + d_row1.z - d_row2.y + d_row2.z,   // v[6] = -d[5] + d[6] - d[9] + d[10]
            d_row1.y - d_row1.w + d_row2.y - d_row2.w     // v[7] = d[5] - d[7] + d[9] - d[11]
        );
        
        // 向量化计算第三组：v[8-11]
        float4 v_row2 = make_float4(
            -d_row1.x + d_row1.z + d_row2.x - d_row2.z,   // v[8] = -d[4] + d[6] + d[8] - d[10]
            -d_row1.y - d_row1.z + d_row2.y + d_row2.z,   // v[9] = -d[5] - d[6] + d[9] + d[10]
            d_row1.y - d_row1.z - d_row2.y + d_row2.z,    // v[10] = d[5] - d[6] - d[9] + d[10]
            -d_row1.y + d_row1.w + d_row2.y - d_row2.w    // v[11] = -d[5] + d[7] + d[9] - d[11]
        );
        
        // 向量化计算第四组：v[12-15]
        float4 v_row3 = make_float4(
            d_row1.x - d_row1.z - d_row3.x + d_row3.z,    // v[12] = d[4] - d[6] - d[12] + d[14]
            d_row1.y + d_row1.z - d_row3.y - d_row3.z,    // v[13] = d[5] + d[6] - d[13] - d[14]
            -d_row1.y + d_row1.z + d_row3.y - d_row3.z,   // v[14] = -d[5] + d[6] + d[13] - d[14]
            d_row1.y - d_row1.w - d_row3.y + d_row3.w     // v[15] = d[5] - d[7] - d[13] + d[15]
        );
        
        // 向量化存储结果
        *((float4*)&v_flat[0]) = v_row0;
        *((float4*)&v_flat[4]) = v_row1;
        *((float4*)&v_flat[8]) = v_row2;
        *((float4*)&v_flat[12]) = v_row3;

        // === 安全的SIMD优化元素级乘法：避免内存对齐问题 ===
        // 安全地加载u和v数据，避免对齐问题
        float4 u_vec0 = make_float4(u_kc[0], u_kc[1], u_kc[2], u_kc[3]);
        float4 u_vec1 = make_float4(u_kc[4], u_kc[5], u_kc[6], u_kc[7]);
        float4 u_vec2 = make_float4(u_kc[8], u_kc[9], u_kc[10], u_kc[11]);
        float4 u_vec3 = make_float4(u_kc[12], u_kc[13], u_kc[14], u_kc[15]);
        
        float4 v_vec0 = *((float4*)&v_flat[0]);   // v_flat是对齐的
        float4 v_vec1 = *((float4*)&v_flat[4]);
        float4 v_vec2 = *((float4*)&v_flat[8]);
        float4 v_vec3 = *((float4*)&v_flat[12]);
        
        float4 m_vec0 = *((float4*)&m_flat[0]);   // m_flat是对齐的
        float4 m_vec1 = *((float4*)&m_flat[4]);
        float4 m_vec2 = *((float4*)&m_flat[8]);
        float4 m_vec3 = *((float4*)&m_flat[12]);
        
        // 向量化融合乘加：一次处理4个元素
        m_vec0 = make_float4(
            fmaf(u_vec0.x, v_vec0.x, m_vec0.x),
            fmaf(u_vec0.y, v_vec0.y, m_vec0.y),
            fmaf(u_vec0.z, v_vec0.z, m_vec0.z),
            fmaf(u_vec0.w, v_vec0.w, m_vec0.w)
        );
        
        m_vec1 = make_float4(
            fmaf(u_vec1.x, v_vec1.x, m_vec1.x),
            fmaf(u_vec1.y, v_vec1.y, m_vec1.y),
            fmaf(u_vec1.z, v_vec1.z, m_vec1.z),
            fmaf(u_vec1.w, v_vec1.w, m_vec1.w)
        );
        
        m_vec2 = make_float4(
            fmaf(u_vec2.x, v_vec2.x, m_vec2.x),
            fmaf(u_vec2.y, v_vec2.y, m_vec2.y),
            fmaf(u_vec2.z, v_vec2.z, m_vec2.z),
            fmaf(u_vec2.w, v_vec2.w, m_vec2.w)
        );
        
        m_vec3 = make_float4(
            fmaf(u_vec3.x, v_vec3.x, m_vec3.x),
            fmaf(u_vec3.y, v_vec3.y, m_vec3.y),
            fmaf(u_vec3.z, v_vec3.z, m_vec3.z),
            fmaf(u_vec3.w, v_vec3.w, m_vec3.w)
        );
        
        // 向量化存储结果
        *((float4*)&m_flat[0]) = m_vec0;
        *((float4*)&m_flat[4]) = m_vec1;
        *((float4*)&m_flat[8]) = m_vec2;
        *((float4*)&m_flat[12]) = m_vec3;
    }

    // === SIMD优化输出变换：使用向量化计算 ===
    // 向量化加载m矩阵
    float4 m_row0 = *((float4*)&m_flat[0]);   // m[0-3]
    float4 m_row1 = *((float4*)&m_flat[4]);   // m[4-7]
    float4 m_row2 = *((float4*)&m_flat[8]);   // m[8-11]
    float4 m_row3 = *((float4*)&m_flat[12]);  // m[12-15]
    
    // 向量化计算输出结果，使用fmaf进行更精确的计算
    float4 Y_vec = make_float4(
        // Y[0] = m[0] + m[1] + m[2] + m[4] + m[5] + m[6] + m[8] + m[9] + m[10]
        fmaf(m_row0.x + m_row0.y + m_row0.z, 1.0f, 
             fmaf(m_row1.x + m_row1.y + m_row1.z, 1.0f, 
                  m_row2.x + m_row2.y + m_row2.z)),
        
        // Y[1] = m[1] - m[2] - m[3] + m[5] - m[6] - m[7] + m[9] - m[10] - m[11]
        fmaf(m_row0.y - m_row0.z - m_row0.w, 1.0f,
             fmaf(m_row1.y - m_row1.z - m_row1.w, 1.0f,
                  m_row2.y - m_row2.z - m_row2.w)),
        
        // Y[2] = m[4] + m[5] + m[6] - m[8] - m[9] - m[10] - m[12] - m[13] - m[14]
        fmaf(m_row1.x + m_row1.y + m_row1.z, 1.0f,
             fmaf(-(m_row2.x + m_row2.y + m_row2.z), 1.0f,
                  -(m_row3.x + m_row3.y + m_row3.z))),
        
        // Y[3] = m[5] - m[6] - m[7] - m[9] + m[10] + m[11] - m[13] + m[14] + m[15]
        fmaf(m_row1.y - m_row1.z - m_row1.w, 1.0f,
             fmaf(-(m_row2.y) + m_row2.z + m_row2.w, 1.0f,
                  -m_row3.y + m_row3.z + m_row3.w))
    );

    // === SIMD优化输出写入：向量化边界检查和写入 ===
    // 预计算输出位置
    int h0 = tile_y * 2;
    int w0 = tile_x * 2;
    int h1 = h0 + 1;
    int w1 = w0 + 1;
    
    // 向量化边界检查
    bool valid[4] = {
        h0 < outH && w0 < outW,  // Y[0]
        h0 < outH && w1 < outW,  // Y[1]
        h1 < outH && w0 < outW,  // Y[2]
        h1 < outH && w1 < outW   // Y[3]
    };
    
    // 条件写入（编译器会优化分支）
    if (valid[0]) output[((n * K + k) * outH + h0) * outW + w0] = Y_vec.x;
    if (valid[1]) output[((n * K + k) * outH + h0) * outW + w1] = Y_vec.y;
    if (valid[2]) output[((n * K + k) * outH + h1) * outW + w0] = Y_vec.z;
    if (valid[3]) output[((n * K + k) * outH + h1) * outW + w1] = Y_vec.w;
}

// 寄存器优化版本的1D Winograd卷积核函数 - 最小化数组使用，最大化寄存器利用
__global__
void winograd_conv_kernel_1D_register_optimized(const float* __restrict__ image,
                                               const float* __restrict__ filter,
                                               float* __restrict__ output,
                                               int N, int C, int H, int W, int K, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    if (idx >= num_tiles) return;

    // 解析线程索引
    int p_local = idx % ((outH / 2) * (outW / 2));
    int k = (idx / ((outH / 2) * (outW / 2))) % K;
    int n = idx / (K * (outH / 2) * (outW / 2));
    int tile_y = p_local / (outW / 2);
    int tile_x = p_local % (outW / 2);

    // 寄存器优化：直接用16个寄存器变量替代数组
    float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
    float m4 = 0.0f, m5 = 0.0f, m6 = 0.0f, m7 = 0.0f;
    float m8 = 0.0f, m9 = 0.0f, m10 = 0.0f, m11 = 0.0f;
    float m12 = 0.0f, m13 = 0.0f, m14 = 0.0f, m15 = 0.0f;

    // 预计算地址偏移，避免在循环中重复计算
    const float* base = image + n * C * H * W;
    int h_start = tile_y * 2;
    int w_start = tile_x * 2;

    // 循环处理所有输入通道
    for (int c = 0; c < C; ++c) {
        // 获取预计算的滤波器变换 - 直接用寄存器变量
        const float* u_kc = filter + (k * C + c) * 16;
        register float u0 = u_kc[0], u1 = u_kc[1], u2 = u_kc[2], u3 = u_kc[3];
        register float u4 = u_kc[4], u5 = u_kc[5], u6 = u_kc[6], u7 = u_kc[7];
        register float u8 = u_kc[8], u9 = u_kc[9], u10 = u_kc[10], u11 = u_kc[11];
        register float u12 = u_kc[12], u13 = u_kc[13], u14 = u_kc[14], u15 = u_kc[15];
        
        // 直接加载输入数据到寄存器，避免数组
        const float* channel_base = base + c * H * W;
        register float d0, d1, d2, d3, d4, d5, d6, d7;
        register float d8, d9, d10, d11, d12, d13, d14, d15;
        
        // 优化的数据加载：展开循环，直接计算地址
        int addr0 = h_start * W + w_start;
        int addr1 = addr0 + 1, addr2 = addr0 + 2, addr3 = addr0 + 3;
        int addr4 = addr0 + W, addr5 = addr4 + 1, addr6 = addr4 + 2, addr7 = addr4 + 3;
        int addr8 = addr0 + 2*W, addr9 = addr8 + 1, addr10 = addr8 + 2, addr11 = addr8 + 3;
        int addr12 = addr0 + 3*W, addr13 = addr12 + 1, addr14 = addr12 + 2, addr15 = addr12 + 3;
        
        // 边界检查和加载 - 使用三元操作符减少分支
        d0 = (h_start >= 0 && h_start < H && w_start >= 0 && w_start < W) ? channel_base[addr0] : 0.0f;
        d1 = (h_start >= 0 && h_start < H && w_start + 1 >= 0 && w_start + 1 < W) ? channel_base[addr1] : 0.0f;
        d2 = (h_start >= 0 && h_start < H && w_start + 2 >= 0 && w_start + 2 < W) ? channel_base[addr2] : 0.0f;
        d3 = (h_start >= 0 && h_start < H && w_start + 3 >= 0 && w_start + 3 < W) ? channel_base[addr3] : 0.0f;
        
        d4 = (h_start + 1 >= 0 && h_start + 1 < H && w_start >= 0 && w_start < W) ? channel_base[addr4] : 0.0f;
        d5 = (h_start + 1 >= 0 && h_start + 1 < H && w_start + 1 >= 0 && w_start + 1 < W) ? channel_base[addr5] : 0.0f;
        d6 = (h_start + 1 >= 0 && h_start + 1 < H && w_start + 2 >= 0 && w_start + 2 < W) ? channel_base[addr6] : 0.0f;
        d7 = (h_start + 1 >= 0 && h_start + 1 < H && w_start + 3 >= 0 && w_start + 3 < W) ? channel_base[addr7] : 0.0f;
        
        d8 = (h_start + 2 >= 0 && h_start + 2 < H && w_start >= 0 && w_start < W) ? channel_base[addr8] : 0.0f;
        d9 = (h_start + 2 >= 0 && h_start + 2 < H && w_start + 1 >= 0 && w_start + 1 < W) ? channel_base[addr9] : 0.0f;
        d10 = (h_start + 2 >= 0 && h_start + 2 < H && w_start + 2 >= 0 && w_start + 2 < W) ? channel_base[addr10] : 0.0f;
        d11 = (h_start + 2 >= 0 && h_start + 2 < H && w_start + 3 >= 0 && w_start + 3 < W) ? channel_base[addr11] : 0.0f;
        
        d12 = (h_start + 3 >= 0 && h_start + 3 < H && w_start >= 0 && w_start < W) ? channel_base[addr12] : 0.0f;
        d13 = (h_start + 3 >= 0 && h_start + 3 < H && w_start + 1 >= 0 && w_start + 1 < W) ? channel_base[addr13] : 0.0f;
        d14 = (h_start + 3 >= 0 && h_start + 3 < H && w_start + 2 >= 0 && w_start + 2 < W) ? channel_base[addr14] : 0.0f;
        d15 = (h_start + 3 >= 0 && h_start + 3 < H && w_start + 3 >= 0 && w_start + 3 < W) ? channel_base[addr15] : 0.0f;
        
        // 寄存器优化的输入变换：直接计算v值，不存储
        // v = B^T @ d @ B 的手动展开版本，直接用于后续计算
        register float v0 = d0 - d2 - d8 + d10;
        register float v1 = d1 + d2 - d9 - d10;
        register float v2 = -d1 + d2 + d9 - d10;
        register float v3 = d1 - d3 - d9 + d11;
        
        register float v4 = d4 - d6 + d8 - d10;
        register float v5 = d5 + d6 + d9 + d10;
        register float v6 = -d5 + d6 - d9 + d10;
        register float v7 = d5 - d7 + d9 - d11;
        
        register float v8 = -d4 + d6 + d8 - d10;
        register float v9 = -d5 - d6 + d9 + d10;
        register float v10 = d5 - d6 - d9 + d10;
        register float v11 = -d5 + d7 + d9 - d11;
        
        register float v12 = d4 - d6 - d12 + d14;
        register float v13 = d5 + d6 - d13 - d14;
        register float v14 = -d5 + d6 + d13 - d14;
        register float v15 = d5 - d7 - d13 + d15;

        // 寄存器优化的元素级乘法：直接累加到m寄存器，使用融合乘加
        m0 = fmaf(u0, v0, m0);
        m1 = fmaf(u1, v1, m1);
        m2 = fmaf(u2, v2, m2);
        m3 = fmaf(u3, v3, m3);
        
        m4 = fmaf(u4, v4, m4);
        m5 = fmaf(u5, v5, m5);
        m6 = fmaf(u6, v6, m6);
        m7 = fmaf(u7, v7, m7);
        
        m8 = fmaf(u8, v8, m8);
        m9 = fmaf(u9, v9, m9);
        m10 = fmaf(u10, v10, m10);
        m11 = fmaf(u11, v11, m11);
        
        m12 = fmaf(u12, v12, m12);
        m13 = fmaf(u13, v13, m13);
        m14 = fmaf(u14, v14, m14);
        m15 = fmaf(u15, v15, m15);
    }

    // 寄存器优化的输出变换：直接计算最终输出，不存储中间结果
    // Y = A^T @ m @ A 的手动展开版本
    register float Y0 = m0 + m1 + m2 + m4 + m5 + m6 + m8 + m9 + m10;
    register float Y1 = m1 - m2 - m3 + m5 - m6 - m7 + m9 - m10 - m11;
    register float Y2 = m4 + m5 + m6 - m8 - m9 - m10 - m12 - m13 - m14;
    register float Y3 = m5 - m6 - m7 - m9 + m10 + m11 - m13 + m14 + m15;

    // 优化的输出写入：预计算输出地址，减少重复计算
    int out_base = (n * K + k) * outH * outW;
    int h0 = tile_y * 2, w0 = tile_x * 2;
    int h1 = h0 + 1, w1 = w0 + 1;
    
    // 条件写入，编译器会优化分支预测
    if (h0 < outH && w0 < outW) output[out_base + h0 * outW + w0] = Y0;
    if (h0 < outH && w1 < outW) output[out_base + h0 * outW + w1] = Y1;
    if (h1 < outH && w0 < outW) output[out_base + h1 * outW + w0] = Y2;
    if (h1 < outH && w1 < outW) output[out_base + h1 * outW + w1] = Y3;
}

void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V, 
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N) {
    const int outH = H - 2;
    const int outW = W - 2;
    
    // Step 1: Precompute filter transformations
    const int threads_per_block_filter = 256;
    int total_filters = K * C;
    int grid_size_filter = (total_filters + threads_per_block_filter - 1) / threads_per_block_filter;
    
    filter_transform_kernel<<<grid_size_filter, threads_per_block_filter>>>(
        filter.data().get(), U.data().get(), K, C
    );
    
    // Step 2: 自适应配置策略 - 根据特征图大小选择最优核函数
    int tiles_x = (outW + 1) / 2;  // X 方向的 tile 数量
    int tiles_y = (outH + 1) / 2;  // Y 方向的 tile 数量
    int tiles_count = tiles_x * tiles_y;
    float sync_ratio = (float)(C * 3) / tiles_count;  // 同步开销与计算量比值

    
            // 线程块尺寸优化 - 尝试不同的线程块配置以提升占用率
    int total_tiles = N * K * tiles_x * tiles_y;
    
    // 根据GPU架构动态选择最优线程块尺寸
    int threads_per_block = 128;  // 减少线程块尺寸，提升寄存器占用率
    int num_blocks = (total_tiles + threads_per_block - 1) / threads_per_block;
        
    // 使用寄存器优化版本，配合优化的线程块尺寸
    winograd_conv_kernel_1D_register_optimized<<<num_blocks, threads_per_block>>>(
         image.data().get(), U.data().get(), out.data().get(),
         N, C, H, W, K, outH, outW
    );
    
    cudaDeviceSynchronize();
}

