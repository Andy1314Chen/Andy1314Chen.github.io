

看 [CUDA Efficient Matrix Transpose 博客](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)的时候，我理解的 simple copy kernel 应该是下面这样的：

```c
__global__ void copy_simple(float *output, const float *input, const int rows, const int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        output[y * cols + x] = input[y * cols + x];
    }
}

// launch kernel
dim3 blockSize(TILE_DIM, TILE_DIM);
dim3 gridSize((nx + TILE_DIM - 1)/TILE_DIM, (ny + TILE_DIM - 1)/TILE_DIM);
copy_simple<<<gridSize, blockSize>>>(d_cdata, d_idata, nx, ny);
```

但博客中的 simple copy kernel 是这样的：

```c

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__ void copy(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = idata[(y + j) * width + x];
}

// launch kernel
dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
```

在 Jetson AGX Orin 64G 上，测试 8192 x 8192 的矩阵，运行结果如下：

```shell
copy_simple kernel execution time: 6.834688 ms
copy kernel execution time: 3.384800 ms
copy shared mem kernel execution time: 3.118784 ms
cudaMemcpyDeviceToDevice execution time: 3.246016 ms
```

速度比我理解的 simple copy kernel 快一倍...，加上 shared memory 优化还能再快一些...

但这里只测了 8192x8192 一个 case，带有 shared memory 优化的 copy kernel 可能性能最好，但大多数情况下应该是官方 `cudaMemcpyDeviceToDevice` 性能要好些，所以一般情况不用自己写 copy kernel。



copy kernel 采用了 tile 的思路进行拷贝，但 kernel 中的逻辑咋一看不是很清晰，例如 stackoverflow 上 [CUDA Matrix Copy - Stack Overflow](https://stackoverflow.com/questions/21371989/cuda-matrix-copy) 、[cuda - Coalesced access in the following matrix copy kernel - Stack Overflow](https://stackoverflow.com/questions/49469506/coalesced-access-in-the-following-matrix-copy-kernel) 等用户的提问，因此记录一下。

copy kernel 将 8192 x 8192 拆分为多个元素数为 32 x 32 个的小 tile，每个 tile 包括 32 x 32 个数据，因此 grid size 为:

```c
dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
```

而在 copy kernel 代码中，每个 thread 要处理 4 个数据（也可以视为又将 tile 分成了 4 个 subtiles），因此 block size 为：

```c
dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
```

```c
// gridDim.x = 32

// blockIdx.x 范围在 (0, TILE_DIM) -- (0, 32)
// blockIdx.y 范围在 (0, TILE_DIM) -- (0, 32)

// threadIdx.x 范围在 (0, TILE_DIM) -- (0, 32)
// threadIdx.y 范围在 (0, BLOCK_ROWS) -- (0, 8)

(y + j)*width + x
    = y * width + j * width + x
	= (blockIdx.y * TILE_DIM) * width + blockIdx.x * TILE_DIM +
	 (j+threadIdx.y) * width + threadIdx.x

// 第一项: (blockIdx.y * TILE_DIM) * width + blockIdx.x * TILE_DIM 等同于矩阵索引为 (blockIdx.y * TILE_DIM, blockIdx.x * TILE_DIM) 的数据，也等同于 (blockIdx.x, blockIdx.y) 所表示 tile 的最上部数据。

// 第二项：(j+threadIdx.y) * width + threadIdx.x,  表示索引为 (threadIdx.x, j + threadIdx.y) 的 tile 中的数据。

// 第一项表示 tile 的全局地址，第二项表示 tile 内部相对地址？？
```


其实主要还是 copy kernel 中符号标识的问题，如果改成下面这个就便于理解了：

```c
__global__ void copy2(float *odata, const float *idata, const int nx=8192) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.x + threadIdx.y;
	int width = nx;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y + j)*width + x] = idata[(y + j)*width + x];
}
```

`x0 = blockIdx.x * blockDim.x, y0 = blockIdx.y * blockDim.x` 是某个 block 左上角数据(x0, y0)的索引，`x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.x + threadIdx.y` 则是指向 subtile 块的，如下图中所示，threadIdx.x 范围在 (0, 32)，threadIdx.y 范围在 (0, 8)。(x0, y0) 再加上 (threadIdx.x， threadIdx.y) 偏移量，整好覆盖了 block 中的第一个 subtile。

前面说到 kernel 的一个 thread block 中共有 32 x 8 个线程，对应一个 subtile 的数据，但每个线程需要处理 4 个数据，所以 copy kernel 中存在一个 for 循环，并且还需要再加上一次偏移量 `(y + j)`，才能覆盖掉下面剩余的 3 个 subtile。



与最简单的 simple_copy_kernel 相比，copy_kernel 多了 tile 的概念，并且一个 thread 要处理 4 个数据，因此在 grid + block 索引的基础上，还要再多一个 j 的偏移量才能访问到正确的数据地址。

![639c9326e5c0a9db0758a225bd66ff3.jpg](https://raw.githubusercontent.com/Andy1314Chen/obsidian-pic/main/image/639c9326e5c0a9db0758a225bd66ff3.jpg)
