# _thomas_kernel
try:
    import cupy as cp
except Exception:
    cp = None
 if cp is not None:   
    _thomas_kernel = cp.RawKernel(r'''


    extern "C" __global__
    void thomas_kernel(
        const float a,
        const float* __restrict__ bvec,
        const float c,
        const float2* __restrict__ d,
        float2* __restrict__ x,
        float2* __restrict__ cprime,
        float2* __restrict__ dprime,
        const int B,
        const int n
    ){
        int k = blockDim.x * blockIdx.x + threadIdx.x;
        if (k >= B) return;

        int base = k * n;
        float b0 = bvec[k];
        float denom = b0;
        float2 d0 = d[base + 0];

        cprime[base + 0] = make_float2(c / denom, 0.0f);
        dprime[base + 0] = make_float2(d0.x / denom, d0.y / denom);

        for (int i = 1; i < n; ++i){
            float2 cp_im1 = cprime[base + (i-1)];
            denom = b0 - a * cp_im1.x;
            cprime[base + i] = make_float2(c / denom, 0.0f);
            float2 di = d[base + i];
            float2 dp_im1 = dprime[base + (i-1)];
            float2 num;
            num.x = di.x - a * dp_im1.x;
            num.y = di.y - a * dp_im1.y;
            dprime[base + i] = make_float2(num.x / denom, num.y / denom);
        }

        x[base + (n-1)] = dprime[base + (n-1)];
        for (int i = n-2; i >= 0; --i){
            float2 dpi = dprime[base + i];
            float2 cpi = cprime[base + i];
            float2 xi1 = x[base + (i+1)];
            float2 val;
            val.x = dpi.x - cpi.x * xi1.x;
            val.y = dpi.y - cpi.x * xi1.y;
            x[base + i] = val;
        }
    }


    ''', 'thomas_kernel')
else:
    _thomas_kernel = None