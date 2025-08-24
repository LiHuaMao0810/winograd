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

__global__
void winograd_conv_kernel_1D(const float* __restrict__ image,
                          const float* __restrict__ filter,
                          float* __restrict__ output,
                          int N, int C, int H, int W, int K, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    if (idx >= num_tiles) return;

    // Decompose thread index to get (n, k, tile_y, tile_x)
    int p_local = idx % ((outH / 2) * (outW / 2));
    int k = (idx / ((outH / 2) * (outW / 2))) % K;
    int n = idx / (K * (outH / 2) * (outW / 2));
    int tile_y = p_local / (outW / 2);
    int tile_x = p_local % (outW / 2);

    float m[4][4] = {{0.0f}};

    // Loop over input channels
    for (int c = 0; c < C; ++c) {
        // --- Load Precomputed Filter Transform ---
        // Note: filter parameter now points to precomputed U matrix
        const float* u_kc = filter + (k * C + c) * 16;
        
        // --- Image Transform ---
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        float d[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int global_h = h_start + i;
                int global_w = w_start + j;
                if (global_h >= 0 && global_h < H && global_w >= 0 && global_w < W) {
                    d[i][j] = image[(n * C + c) * H * W + global_h * W + global_w];
                } else {
                    d[i][j] = 0.0f;  // Zero padding
                }
            }
        }
        float v_ncp[4][4];
        float temp_d[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                temp_d[i][j] = B_T[i][0] * d[0][j] + B_T[i][1] * d[1][j] + B_T[i][2] * d[2][j] + B_T[i][3] * d[3][j];
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                v_ncp[i][j] = temp_d[i][0] * B[0][j] + temp_d[i][1] * B[1][j] + temp_d[i][2] * B[2][j] + temp_d[i][3] * B[3][j];
            }
        }

        // --- Element-wise product and accumulate ---
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[i][j] += u_kc[i * 4 + j] * v_ncp[i][j];
            }
        }
    }

    // --- Output Transform ---
    float temp_m[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            temp_m[i][j] = A_T[i][0] * m[0][j] + A_T[i][1] * m[1][j] + A_T[i][2] * m[2][j] + A_T[i][3] * m[3][j];
        }
    }
    float Y[2][2];
    for (int i = 0; i < 2; ++i) {
        Y[i][0] = temp_m[i][0] + temp_m[i][1] + temp_m[i][2];
        Y[i][1] = temp_m[i][1] - temp_m[i][2] - temp_m[i][3];
    }

    // --- Write output ---
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int h = tile_y * 2 + i;
            int w = tile_x * 2 + j;
            if (h < outH && w < outW) {
                output[((n * K + k) * outH + h) * outW + w] = Y[i][j];
            }
        }
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
            
            // --- 协作加载变换后的卷积核到共享内存 ---
            for (int load_idx = tid; load_idx < filter_shared_size; load_idx += total_threads) {
                int k_local = load_idx / 16;  // 块内的输出通道索引
                int filter_elem = load_idx % 16;  // 4x4矩阵中的元素索引
                int k_global = blockIdx.z * blockDim.z + k_local;  // 全局输出通道索引
                
                if (k_global < K) {
                    const float* u_kc = filter + (k_global * C + c) * 16;
                    shared_filters[load_idx] = u_kc[filter_elem];
                } else {
                    shared_filters[load_idx] = 0.0f;
                }
            }
            
            __syncthreads();
            
            // --- 协作加载输入数据到共享内存 ---
            // 计算当前块需要的输入数据范围
            int input_start_h = blockIdx.y * blockDim.y * 2;  // 块的起始高度
            int input_start_w = blockIdx.x * blockDim.x * 2;  // 块的起始宽度
            
            for (int load_idx = tid; load_idx < input_shared_size; load_idx += total_threads) {
                int local_h = load_idx / input_tile_w;
                int local_w = load_idx % input_tile_w;
                int global_h = input_start_h + local_h;
                int global_w = input_start_w + local_w;
                
                // 边界检查和加载数据
                if (global_h >= 0 && global_h < H && global_w >= 0 && global_w < W) {
                    shared_input[load_idx] = image[(n * C + c) * H * W + global_h * W + global_w];
                    // shared_input[load_idx] = image[n][c][global_h][global_w]
                } else {
                    shared_input[load_idx] = 0.0f;  // Zero padding
                }
            }
            
            __syncthreads();
            
            // --- 从共享内存进行Winograd变换和计算 ---
            // 计算当前线程对应的输入数据在共享内存中的位置
            int local_h_start = threadIdx.y * 2;
            int local_w_start = threadIdx.x * 2;
            
            // 确保访问不越界，且当前线程在有效范围内
            if (tile_x < tiles_x && tile_y < tiles_y && k < K && 
                local_h_start + 3 < input_tile_h && local_w_start + 3 < input_tile_w) {
                // 提取4x4输入块
                float temp[16];
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        temp[i * 4 + j] = shared_input[(local_h_start + i) * input_tile_w + (local_w_start + j)];
                        // temp[i][j] = shared_input[2*y+i][2*x+j]
                    }
                }
                
                // 完整的输入变换: v = B^T @ temp @ B
                float temp1[16];  // B^T @ temp 的结果
                float v[16];      // B^T @ temp @ B 的最终结果
                
                // 步骤 1: temp1 = B^T @ temp
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        temp1[i * 4 + j] = 
                            B_T[i][0] * temp[0 * 4 + j] +
                            B_T[i][1] * temp[1 * 4 + j] +
                            B_T[i][2] * temp[2 * 4 + j] +
                            B_T[i][3] * temp[3 * 4 + j];
                    }
                }
                
                // 步骤 2: v = temp1 @ B
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        v[i * 4 + j] = 
                            temp1[i * 4 + 0] * B[0][j] +
                            temp1[i * 4 + 1] * B[1][j] +
                            temp1[i * 4 + 2] * B[2][j] +
                            temp1[i * 4 + 3] * B[3][j];
                    }
                }
                
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

        // --- 输出变换 ---（只有有效线程进行输出变换和写入）
        if (tile_x < tiles_x && tile_y < tiles_y && k < K) {
            // 计算 Y = A^T @ accumulator @ A
            // 步骤 1: temp_out = A^T @ accumulator
            float temp_out[8]; // 2x4 结果
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    temp_out[i * 4 + j] = 
                        A_T[i][0] * accumulator[0 * 4 + j] +
                        A_T[i][1] * accumulator[1 * 4 + j] +
                        A_T[i][2] * accumulator[2 * 4 + j] +
                        A_T[i][3] * accumulator[3 * 4 + j];
                }
            }
            
            // 步骤 2: Y = temp_out @ A，其中A = A_T^T (A_T的转置)
            // A_T = [[1,1,1,0], [0,1,-1,-1]], 所以 A = [[1,0], [1,1], [1,-1], [0,-1]]
            float Y[4]; // 2x2 最终输出
            for (int i = 0; i < 2; ++i) {
                // 第一列 (j=0): A[:,0] = [1,1,1,0]
                Y[i * 2 + 0] = temp_out[i * 4 + 0] * 1.0f + temp_out[i * 4 + 1] * 1.0f + 
                               temp_out[i * 4 + 2] * 1.0f + temp_out[i * 4 + 3] * 0.0f;
                // 第二列 (j=1): A[:,1] = [0,1,-1,-1]  
                Y[i * 2 + 1] = temp_out[i * 4 + 0] * 0.0f + temp_out[i * 4 + 1] * 1.0f + 
                               temp_out[i * 4 + 2] * (-1.0f) + temp_out[i * 4 + 3] * (-1.0f);
            }
            
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

    if (H * W <= 30*30 || sync_ratio > 8.0f) {
        // 小特征图或同步开销过大：使用1D无共享内存核函数
        printf("Using 1D kernel (small feature map or high sync overhead)\n");
        
        int total_work = N * K * tiles_x * tiles_y;
        int threads_per_block = 256;
        int num_blocks = (total_work + threads_per_block - 1) / threads_per_block;
        
        winograd_conv_kernel_1D<<<num_blocks, threads_per_block>>>(
            image.data().get(), U.data().get(), out.data().get(),
            N, C, H, W, K, outH, outW
        );
        
    } else {
        // 大特征图：使用3D共享内存核函数
        printf("Using 3D kernel (large feature map, shared memory beneficial)\n");
        
        // 根据特征图大小调整块配置
        dim3 blockDim;
        if (H * W > 80 * 80) {
            // 超大特征图：优先空间并行
            blockDim = dim3(8, 8, 8);
        } else {
            // 中大特征图：平衡空间和通道并行  
            blockDim = dim3(4, 4, 16);
        }
        
        dim3 gridDim(
            (tiles_x + blockDim.x - 1) / blockDim.x,
            (tiles_y + blockDim.y - 1) / blockDim.y,  
            (K + blockDim.z - 1) / blockDim.z
        );
        
        // 计算共享内存大小
        int input_tile_h = blockDim.y * 2 + 2;
        int input_tile_w = blockDim.x * 2 + 2;
        int input_shared_size = input_tile_h * input_tile_w;
        int filter_shared_size = 16 * blockDim.z;
        size_t shared_memory_size = (input_shared_size + filter_shared_size) * sizeof(float);

        winograd_conv_kernel<<<gridDim, blockDim, shared_memory_size>>>(
            image.data().get(), U.data().get(), out.data().get(),
            N, C, H, W, K, outH, outW
        );
    }

    cudaDeviceSynchronize();
}