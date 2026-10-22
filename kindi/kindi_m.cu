/* kindi.cu - Kindi 0.1.0 CUDA kernels (sm_30 / Quadro K420). Compiled at run time by NVRTC (CuPy).
   Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md. */

/* ---------------------------------------------------------------- helpers */

__device__ __forceinline__ float h2f(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }

__device__ __forceinline__ float hf(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }

__device__ __forceinline__ float h2f3(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }

__device__ __forceinline__ double wsum_d(double v){
  for(int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}

__device__ __forceinline__ float wsum_f(float v){
  for(int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}

__device__ __forceinline__ void fwht_chunks(float* xs, int NC, int j){
  for(int h = 1; h < 512; h <<= 1){
    for(int c = 0; c < NC; c += 2){
      const int cc = c + (j >> 8);
      if(cc < NC){
        const int jj = j & 255;
        const int i = (jj / h) * 2 * h + (jj % h) + (cc << 9);
        const float a = xs[i], b = xs[i + h];
        xs[i] = a + b; xs[i + h] = a - b;
      }
    }
    __syncthreads();
  }
}


/* ---------------------------------------------------------------- ternary GEMV (decode): k_g4 / k_g4l */

#define RB 16
#define GROUPS_PER_BLOCK 32

struct Grp { const uint4* W; const unsigned int* S; float* y; int R; int g; };

__device__ __forceinline__ Grp grp(int g, int NG0, int NG1,
    const uint4* W0, const unsigned int* S0, float* y0, int R0,
    const uint4* W1, const unsigned int* S1, float* y1, int R1,
    const uint4* W2, const unsigned int* S2, float* y2, int R2)
{
  Grp q; q.W = W0; q.S = S0; q.y = y0; q.R = R0;
  if (g >= NG0) {
    g -= NG0; q.W = W1; q.S = S1; q.y = y1; q.R = R1;
    if (g >= NG1) {
      g -= NG1; q.W = W2; q.S = S2; q.y = y2; q.R = R2;
    }
  }
  q.g = g;
  return q;
}

extern "C" __global__ void __launch_bounds__(512) k_g4(
    const uint4* __restrict__ W0, const unsigned int* __restrict__ S0, float* __restrict__ y0, int R0,
    const uint4* __restrict__ W1, const unsigned int* __restrict__ S1, float* __restrict__ y1, int R1,
    const uint4* __restrict__ W2, const unsigned int* __restrict__ S2, float* __restrict__ y2, int R2,
    const float* __restrict__ x, int C)
{
  __shared__ float T[2 * 81 * 64];
  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nw   = blockDim.x >> 5;

  const int CW  = C >> 4;
  const int NPH = C >> 9;
  const int NG0 = R0 >> 4, NG1 = R1 >> 4;
  const int NG  = NG0 + NG1 + (R2 >> 4);

  const int g0 = blockIdx.x * GROUPS_PER_BLOCK;
  const int g1 = min(g0 + GROUPS_PER_BLOCK, NG);

  const char* Tc = (const char*)T;
  const unsigned int l4 = lane << 2;
  const bool b0 = lane & 1, b1 = lane & 2, b2 = lane & 4, b3 = lane & 8;
  const int rfin = (b0 ? 8 : 0) + (b1 ? 4 : 0) + (b2 ? 2 : 0) + (b3 ? 1 : 0);
  const bool wr = (lane & 16) == 0;

  const int bl = tid & 31, bk = (tid >> 5) & 3;
  float nx0 = 0.f, nx1 = 0.f, nx2 = 0.f, nx3 = 0.f;
  if (tid < 384) {
    const float* xp = x + (bl << 4) + (bk << 2);
    nx0 = xp[0]; nx1 = xp[1]; nx2 = xp[2]; nx3 = xp[3];
  }

  for (int ph = 0; ph < NPH; ++ph) {
    __syncthreads();
    if (tid < 384) {
      const int l = bl, k = bk, t3 = tid >> 7;
      const float x0 = nx0, x1 = nx1, x2 = nx2, x3 = nx3;
      if (ph + 1 < NPH) {
        const float* xq = x + ((ph + 1) << 9) + (l << 4) + (k << 2);
        nx0 = xq[0]; nx1 = xq[1]; nx2 = xq[2]; nx3 = xq[3];
      }
      float* Tk = T + (k >> 1) * 5184 + (k & 1) * 32 + l + t3 * 27 * 64;
      const float bb3 = (float)(t3 - 1) * x3;
#pragma unroll
      for (int t2 = 0; t2 < 3; ++t2) {
        const float bb2 = bb3 + (float)(t2 - 1) * x2;
#pragma unroll
        for (int t1 = 0; t1 < 3; ++t1) {
          const float bb1 = bb2 + (float)(t1 - 1) * x1;
          float* q = Tk + (t2 * 9 + t1 * 3) * 64;
          q[0]   = bb1 - x0;
          q[64]  = bb1;
          q[128] = bb1 + x0;
        }
      }
    }
    __syncthreads();

    for (int g = g0 + warp; g < g1; g += nw) {
      const Grp q = grp(g, NG0, NG1,
                        W0, S0, y0, R0,
                        W1, S1, y1, R1,
                        W2, S2, y2, R2);
      const int rw = (q.g << 4) + rfin;
      const float yo = (ph && wr) ? q.y[rw] : 0.0f;

      uint4 wv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        wv[i] = q.W[(long long)((q.g << 2) + i) * CW + (ph << 5) + lane];

      const unsigned int sv = q.S[((long long)q.g * NPH + ph) * 32 + lane];

      float v[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const unsigned int ww[4] = { wv[i].x, wv[i].y, wv[i].z, wv[i].w };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const unsigned int w = ww[j];
          v[4 * i + j] =
              (*(const float*)(Tc + __byte_perm(w, l4, 0x5504u))
             + *(const float*)(Tc + 128 + __byte_perm(w, l4, 0x5514u)))
            + (*(const float*)(Tc + 20736 + __byte_perm(w, l4, 0x5524u))
             + *(const float*)(Tc + 20864 + __byte_perm(w, l4, 0x5534u)));
        }
      }

      float a8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        a8[i] = (b0 ? v[8 + i] : v[i]) + __shfl_xor(b0 ? v[i] : v[8 + i], 1);

      float a4[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        a4[i] = (b1 ? a8[4 + i] : a8[i]) + __shfl_xor(b1 ? a8[i] : a8[4 + i], 2);

      float a2[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        a2[i] = (b2 ? a4[2 + i] : a4[i]) + __shfl_xor(b2 ? a4[i] : a4[2 + i], 4);

      a2[0] *= h2f((unsigned short)(sv & 0xFFFFu));
      a2[1] *= h2f((unsigned short)(sv >> 16));

      float d = (b3 ? a2[1] : a2[0]) + __shfl_xor(b3 ? a2[0] : a2[1], 8);
      d += __shfl_xor(d, 16);

      if (wr) q.y[rw] = yo + d;
    }
  }
}

/* shortlist head: k_g4 over a list of 16-row groups, compact output + token id per slot */

/* struct Grp and grp() are already defined above k_g4; reuse them.
   k_g4l does not need grp() because it indexes through glist. */

extern "C" __global__ void __launch_bounds__(512) k_g4l(
    const uint4* __restrict__ W0, const unsigned int* __restrict__ S0, float* __restrict__ y0, int R0,
    const uint4* __restrict__ W1, const unsigned int* __restrict__ S1, float* __restrict__ y1, int R1,
    const uint4* __restrict__ W2, const unsigned int* __restrict__ S2, float* __restrict__ y2, int R2,
    const float* __restrict__ x, int C,
    const int* __restrict__ glist, int ngl,
    const int* __restrict__ permtok, int* __restrict__ tokc)
{
  __shared__ float T[2 * 81 * 64];
  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nw   = blockDim.x >> 5;

  const int CW  = C >> 4;
  const int NPH = C >> 9;
  const int NG  = ngl;

  const int g0 = blockIdx.x * GROUPS_PER_BLOCK;
  const int g1 = min(g0 + GROUPS_PER_BLOCK, NG);

  const char* Tc = (const char*)T;
  const unsigned int l4 = lane << 2;
  const bool b0 = lane & 1, b1 = lane & 2, b2 = lane & 4, b3 = lane & 8;
  const int rfin = (b0 ? 8 : 0) + (b1 ? 4 : 0) + (b2 ? 2 : 0) + (b3 ? 1 : 0);
  const bool wr = (lane & 16) == 0;

  const int bl = tid & 31, bk = (tid >> 5) & 3;
  float nx0 = 0.f, nx1 = 0.f, nx2 = 0.f, nx3 = 0.f;
  if (tid < 384) {
    const float* xp = x + (bl << 4) + (bk << 2);
    nx0 = xp[0]; nx1 = xp[1]; nx2 = xp[2]; nx3 = xp[3];
  }

  for (int ph = 0; ph < NPH; ++ph) {
    __syncthreads();
    if (tid < 384) {
      const int l = bl, k = bk, t3 = tid >> 7;
      const float x0 = nx0, x1 = nx1, x2 = nx2, x3 = nx3;
      if (ph + 1 < NPH) {
        const float* xq = x + ((ph + 1) << 9) + (l << 4) + (k << 2);
        nx0 = xq[0]; nx1 = xq[1]; nx2 = xq[2]; nx3 = xq[3];
      }
      float* Tk = T + (k >> 1) * 5184 + (k & 1) * 32 + l + t3 * 27 * 64;
      const float bb3 = (float)(t3 - 1) * x3;
#pragma unroll
      for (int t2 = 0; t2 < 3; ++t2) {
        const float bb2 = bb3 + (float)(t2 - 1) * x2;
#pragma unroll
        for (int t1 = 0; t1 < 3; ++t1) {
          const float bb1 = bb2 + (float)(t1 - 1) * x1;
          float* q = Tk + (t2 * 9 + t1 * 3) * 64;
          q[0]   = bb1 - x0;
          q[64]  = bb1;
          q[128] = bb1 + x0;
        }
      }
    }
    __syncthreads();

    for (int g = g0 + warp; g < g1; g += nw) {
      Grp q;
      q.W = W0; q.S = S0; q.y = y0; q.R = R0;
      q.g = glist[g];

      const int rw = (g << 4) + rfin;   /* compact slot in this launch's output */
      if (ph == 0 && wr)
        tokc[rw] = permtok[(q.g << 4) + rfin];

      const float yo = (ph && wr) ? q.y[rw] : 0.0f;

      uint4 wv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        wv[i] = q.W[(long long)((q.g << 2) + i) * CW + (ph << 5) + lane];

      const unsigned int sv = q.S[((long long)q.g * NPH + ph) * 32 + lane];

      float v[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const unsigned int ww[4] = { wv[i].x, wv[i].y, wv[i].z, wv[i].w };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const unsigned int w = ww[j];
          v[4 * i + j] =
              (*(const float*)(Tc + __byte_perm(w, l4, 0x5504u))
             + *(const float*)(Tc + 128 + __byte_perm(w, l4, 0x5514u)))
            + (*(const float*)(Tc + 20736 + __byte_perm(w, l4, 0x5524u))
             + *(const float*)(Tc + 20864 + __byte_perm(w, l4, 0x5534u)));
        }
      }

      float a8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        a8[i] = (b0 ? v[8 + i] : v[i]) + __shfl_xor(b0 ? v[i] : v[8 + i], 1);

      float a4[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        a4[i] = (b1 ? a8[4 + i] : a8[i]) + __shfl_xor(b1 ? a8[i] : a8[4 + i], 2);

      float a2[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        a2[i] = (b2 ? a4[2 + i] : a4[i]) + __shfl_xor(b2 ? a4[i] : a4[2 + i], 4);

      a2[0] *= h2f((unsigned short)(sv & 0xFFFFu));
      a2[1] *= h2f((unsigned short)(sv >> 16));

      float d = (b3 ? a2[1] : a2[0]) + __shfl_xor(b3 ? a2[0] : a2[1], 8);
      d += __shfl_xor(d, 16);

      if (wr) q.y[rw] = yo + d;
    }
  }
}

/* ---------------------------------------------------------------- norms, folds, GDN, RoPE */

extern "C" __global__ void k_fold_fwht(const float* __restrict__ x, const float* __restrict__ sg,
                            float* __restrict__ xo){
  __shared__ float xs[512];
  const int c = blockIdx.x, j = threadIdx.x;
  xs[j] = sg[(c << 9) + j] * x[(c << 9) + j];
  __syncthreads();
  for(int h = 1; h < 512; h <<= 1){
    if(j < 256){
      const int i = (j / h) * 2 * h + (j % h);
      const float a = xs[i], b = xs[i + h];
      xs[i] = a + b; xs[i + h] = a - b;
    }
    __syncthreads();
  }
  xo[(c << 9) + j] = xs[j] * 0.044194173824159220878f;
}

extern "C" __global__ void k_rmsfold(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ sg,
                          float* __restrict__ xn, float* __restrict__ xo){
  __shared__ float xs[2048];
  __shared__ double red[16];
  __shared__ float bscale;
  const int j = threadIdx.x, lane = j & 31, wp = j >> 5;
  double acc = 0.0;
  for(int e = j; e < 2048; e += 512){ const double v = (double)x[e]; acc += v * v; }
  acc = wsum_d(acc);
  if(lane == 0) red[wp] = acc;
  __syncthreads();
  if(wp == 0){
    double t = lane < 16 ? red[lane] : 0.0;
    t = wsum_d(t);
    if(lane == 0) bscale = 1.0f / sqrtf((float)(t / 2048) + 1e-6f);
  }
  __syncthreads();
  const float s = bscale;
  for(int e = j; e < 2048; e += 512){
    const float v = (x[e] * s) * w[e];
    xn[e] = v;
    xs[e] = sg[e] * v;
  }
  __syncthreads();
  fwht_chunks(xs, 4, j);
  for(int e = j; e < 2048; e += 512) xo[e] = xs[e] * 0.044194173824159220878f;
}

extern "C" __global__ void k_smfold(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ sg,
                         float* __restrict__ xo){
  __shared__ float xs[512];
  const int c = blockIdx.x, j = threadIdx.x;
  const int e = (c << 9) + j;
  const float xa = a[e];
  const float m = (xa / (1.0f + expf(-xa))) * b[e];
  xs[j] = sg[e] * m;
  __syncthreads();
  fwht_chunks(xs, 1, j);
  xo[e] = xs[j] * 0.044194173824159220878f;
}

extern "C" __global__ void k_l2w(const float* __restrict__ x, float* __restrict__ out, int n, int rows){
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if(r >= rows) return;
  const float* xr = x + (long long)r * n;
  double acc = 0.0;
  for(int e = lane; e < n; e += 32){ const double v = (double)xr[e]; acc += v * v; }
  acc = wsum_d(acc);
  const float sc = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
  for(int e = lane; e < n; e += 32) out[(long long)r * n + e] = xr[e] * sc;
}

extern "C" __global__ void k_rmsw(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int n, int rows){
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if(r >= rows) return;
  const float* xr = x + (long long)r * n;
  double acc = 0.0;
  for(int e = lane; e < n; e += 32){ const double v = (double)xr[e]; acc += v * v; }
  acc = wsum_d(acc);
  const float sc = 1.0f / sqrtf((float)(acc / n) + 1e-6f);
  for(int e = lane; e < n; e += 32) out[(long long)r * n + e] = (xr[e] * sc) * w[e];
}

extern "C" __global__ void k_gdn_gates(const float* __restrict__ xn, const float* __restrict__ bw,
                            const float* __restrict__ aw, const float* __restrict__ dtb,
                            const float* __restrict__ ssma, float* __restrict__ beta,
                            float* __restrict__ g, int n){
  __shared__ float sb[256];
  __shared__ float sa[256];
  const int h = blockIdx.x, t = threadIdx.x;
  float ab = 0.f, aa = 0.f;
  for(int i = t; i < n; i += 256){
    const float x = xn[i];
    ab += bw[(long long)h * n + i] * x;
    aa += aw[(long long)h * n + i] * x;
  }
  sb[t] = ab; sa[t] = aa;
  __syncthreads();
  for(int st = 128; st > 0; st >>= 1){
    if(t < st){ sb[t] += sb[t + st]; sa[t] += sa[t + st]; }
    __syncthreads();
  }
  if(t == 0){
    beta[h] = 1.0f / (1.0f + expf(-sb[0]));
    const float a = sa[0] + dtb[h];
    const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
    g[h] = sp * ssma[h];
  }
}

extern "C" __global__ void k_gdn_conv(const float* __restrict__ qkv, float* __restrict__ conv,
                           const float* __restrict__ cw, float* __restrict__ cs, int C){
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= C) return;
  const float c0 = conv[c], c1 = conv[C + c], c2 = conv[2 * C + c], nw = qkv[c];
  const float o = c0 * cw[4 * c] + c1 * cw[4 * c + 1] + c2 * cw[4 * c + 2] + nw * cw[4 * c + 3];
  conv[c] = c1; conv[C + c] = c2; conv[2 * C + c] = nw;
  cs[c] = o / (1.0f + expf(-o));
}

extern "C" __global__ void k_gdn_delta(float* __restrict__ M, const float* __restrict__ q,
                            const float* __restrict__ k, const float* __restrict__ v,
                            const float* __restrict__ g, const float* __restrict__ beta,
                            float* __restrict__ out, float scale){
  const int h = blockIdx.x, j = threadIdx.x;          /* head, value column (128) */
  __shared__ float ks[128];
  __shared__ float qs[128];
  ks[j] = k[h * 128 + j];
  qs[j] = q[h * 128 + j];
  __syncthreads();
  const float e = expf(g[h]);
  float* Mh = M + (long long)h * 128 * 128;
  float sk = 0.f;
  for(int i = 0; i < 128; ++i) sk += (Mh[i * 128 + j] * e) * ks[i];
  const float dd = (v[h * 128 + j] - sk) * beta[h];
  float o = 0.f;
  for(int i = 0; i < 128; ++i){
    const float m = Mh[i * 128 + j] * e + ks[i] * dd;
    Mh[i * 128 + j] = m;
    o += m * qs[i];
  }
  out[h * 128 + j] = o * scale;
}

extern "C" __global__ void k_rope(const float* __restrict__ x,
                       const float* __restrict__ cc,
                       const float* __restrict__ ss,
                       float* __restrict__ out, int HD){
  const int r = blockIdx.x;
  const int j = threadIdx.x;
  const float* xr = x + (long long)r * HD;
  float* orow = out + (long long)r * HD;
  if(j < 32){
    const float a = xr[j], b = xr[j + 32];
    orow[j]      = a * cc[j] - b * ss[j];
    orow[j + 32] = a * ss[j] + b * cc[j];
  } else if(j >= 64){
    orow[j] = xr[j];
  }
}

extern "C" __global__ void k_deq_emb3(const unsigned char* __restrict__ bc, const unsigned short* __restrict__ dpb,
                           const float* __restrict__ sg, float* __restrict__ h, int row){
  __shared__ float w[2048];
  const int t = threadIdx.x;
  const long long r4 = row >> 2, rl = row & 3, g16 = row >> 4;
  const int rr = row & 15;
  const int lane_low = ((rr >> 3) & 1) | (((rr >> 2) & 1) << 1) | (((rr >> 1) & 1) << 2), jj = rr & 1;
  for(int e = t; e < 2048; e += 1024){
    const int s = e >> 2, p = e >> 7, r = e & 3;
    const unsigned int b = bc[(r4 * 128 + (s >> 2)) * 16 + rl * 4 + (s & 3)];
    const int dig = (r == 0) ? (b % 3) : ((r == 1) ? ((b / 3) % 3) : ((r == 2) ? ((b / 9) % 3) : ((b / 27) % 3)));
    const int lane = ((p & 3) << 3) | lane_low;
    w[e] = ((float)(dig - 1)) * h2f3(dpb[((g16 * 4 + (p >> 2)) * 32 + lane) * 2 + jj]);
  }
  __syncthreads();
  for(int j2 = t; j2 < 2048; j2 += 1024){
    const int c = j2 >> 9, j = j2 & 511;
    float acc = 0.f;
#pragma unroll 8
    for(int i = 0; i < 512; ++i){
      const float ww = w[(c << 9) + i];
      acc += (((__popc(i & j) & 1) == 1) ? -ww : ww);
    }
    h[j2] = sg[j2] * (acc * 0.044194173824159220878f);
  }
}


/* ---------------------------------------------------------------- attention (fp16 K/V, GQA 4:1) */

#define ATV_CH 256

extern "C" __global__ void k_attn_s4w(const uint4* __restrict__ Kh, const float* __restrict__ q, float* __restrict__ s, int n){
  /* grid (NHKV, ceil(n/32)), block 256: warp = 4 consecutive positions; lane = dims 8l..8l+7 with the 4 heads' q
     in registers (loaded once); one coalesced 16-byte K load per lane per position; the 16 sums (4 positions x 4
     heads) reduced by a straight-line multi-value butterfly (16 shuffles per 4 positions) */
  const int kv = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.y * 8 + warp, nwarps = gridDim.y * 8;     /* persistent: q loaded once per warp */
  if(gw * 4 >= n) return;
  float qr[4][8];
#pragma unroll
  for(int j = 0; j < 4; ++j){
    const float4 a = ((const float4*)(q + kv * 1024 + j * 256 + lane * 8))[0];
    const float4 b = ((const float4*)(q + kv * 1024 + j * 256 + lane * 8))[1];
    qr[j][0] = a.x; qr[j][1] = a.y; qr[j][2] = a.z; qr[j][3] = a.w;
    qr[j][4] = b.x; qr[j][5] = b.y; qr[j][6] = b.z; qr[j][7] = b.w;
  }
  for(int t0 = gw * 4; t0 < n; t0 += nwarps * 4){
  float v[16];
#pragma unroll
  for(int p = 0; p < 4; ++p){
    const int t = min(t0 + p, n - 1);
    const uint4 w = Kh[((long long)t * 2 + kv) * 32 + lane];
    const unsigned int ww[4] = {w.x, w.y, w.z, w.w};
    float kk[8];
#pragma unroll
    for(int m = 0; m < 4; ++m){ kk[2 * m] = hf((unsigned short)(ww[m] & 0xFFFFu)); kk[2 * m + 1] = hf((unsigned short)(ww[m] >> 16)); }
#pragma unroll
    for(int j = 0; j < 4; ++j){
      float a = 0.f;
#pragma unroll
      for(int i = 0; i < 8; ++i) a += kk[i] * qr[j][i];
      v[p * 4 + j] = a;
    }
  }
  { const bool h0 = (lane & 16) != 0; const float b0_0 = (h0 ? v[8] : v[0]) + __shfl_xor(h0 ? v[0] : v[8], 16); const float b0_1 = (h0 ? v[9] : v[1]) + __shfl_xor(h0 ? v[1] : v[9], 16); const float b0_2 = (h0 ? v[10] : v[2]) + __shfl_xor(h0 ? v[2] : v[10], 16); const float b0_3 = (h0 ? v[11] : v[3]) + __shfl_xor(h0 ? v[3] : v[11], 16); const float b0_4 = (h0 ? v[12] : v[4]) + __shfl_xor(h0 ? v[4] : v[12], 16); const float b0_5 = (h0 ? v[13] : v[5]) + __shfl_xor(h0 ? v[5] : v[13], 16); const float b0_6 = (h0 ? v[14] : v[6]) + __shfl_xor(h0 ? v[6] : v[14], 16); const float b0_7 = (h0 ? v[15] : v[7]) + __shfl_xor(h0 ? v[7] : v[15], 16); const bool h1 = (lane & 8) != 0; const float b1_0 = (h1 ? b0_4 : b0_0) + __shfl_xor(h1 ? b0_0 : b0_4, 8); const float b1_1 = (h1 ? b0_5 : b0_1) + __shfl_xor(h1 ? b0_1 : b0_5, 8); const float b1_2 = (h1 ? b0_6 : b0_2) + __shfl_xor(h1 ? b0_2 : b0_6, 8); const float b1_3 = (h1 ? b0_7 : b0_3) + __shfl_xor(h1 ? b0_3 : b0_7, 8); const bool h2 = (lane & 4) != 0; const float b2_0 = (h2 ? b1_2 : b1_0) + __shfl_xor(h2 ? b1_0 : b1_2, 4); const float b2_1 = (h2 ? b1_3 : b1_1) + __shfl_xor(h2 ? b1_1 : b1_3, 4); const bool h3 = (lane & 2) != 0; const float b3_0 = (h3 ? b2_1 : b2_0) + __shfl_xor(h3 ? b2_0 : b2_1, 2); const float b4_0 = b3_0 + __shfl_xor(b3_0, 1); v[0] = b4_0; }
  const int idx = ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
  const int t = t0 + (idx >> 2), j = idx & 3;
  if((lane & 1) == 0 && t < n) s[(kv * 4 + j) * n + t] = v[0] * 0.0625f;
  }
}

extern "C" __global__ void k_softmax_rows(float* __restrict__ s, int n){
  /* grid NH, block 256: max-subtracted softmax over the n positions of each head, in place */
  const int hh = blockIdx.x, t = threadIdx.x;
  float* sr = s + (long long)hh * n;
  __shared__ float red[256];
  __shared__ float pmax;
  float mx = -1e30f;
  for(int i = t; i < n; i += 256) mx = fmaxf(mx, sr[i]);
  red[t] = mx;
  __syncthreads();
  for(int st = 128; st > 0; st >>= 1){
    if(t < st) red[t] = fmaxf(red[t], red[t + st]);
    __syncthreads();
  }
  if(t == 0) pmax = red[0];
  __syncthreads();
  const float m = pmax;
  float acc0 = 0.0f;
  for(int i = t; i < n; i += 256) acc0 += expf(sr[i] - m);
  __syncthreads();
  red[t] = acc0;
  __syncthreads();
  for(int st = 128; st > 0; st >>= 1){
    if(t < st) red[t] += red[t + st];
    __syncthreads();
  }
  const float tot = red[0];
  __syncthreads();
  for(int i = t; i < n; i += 256) sr[i] = expf(sr[i] - m) / tot;
}

extern "C" __global__ void k_attn_v4p(const float* __restrict__ s, const unsigned int* __restrict__ Vh2,
                           float* __restrict__ part, int n){
  /* grid (NHKV, 1, nchunks), block 128: thread = features 2f, 2f+1 (one half2 load), 4 heads; partial sums.
     The 4 heads' probabilities of a position are one float4 in shared memory (one broadcast load). */
  __shared__ float4 sp4[ATV_CH];
  const int kv = blockIdx.x, f2 = threadIdx.x, c = blockIdx.z;
  const int i0 = c * ATV_CH, m = min(ATV_CH, n - i0);
  for(int i = threadIdx.x; i < ATV_CH; i += blockDim.x){
    float4 p4 = make_float4(0.f, 0.f, 0.f, 0.f);
    if(i < m){
      const float* s0 = s + (long long)(kv * 4) * n + i0 + i;
      p4 = make_float4(s0[0], s0[n], s0[2 * n], s0[3 * n]);
    }
    sp4[i] = p4;
  }
  __syncthreads();
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
  const unsigned int* vb = Vh2 + ((long long)i0 * 2 + kv) * 128 + f2;
#pragma unroll 8
  for(int i = 0; i < m; ++i){
    const unsigned int vv = vb[(long long)i * 256];
    const float v0 = hf((unsigned short)(vv & 0xFFFFu)), v1 = hf((unsigned short)(vv >> 16));
    const float4 p = sp4[i];
    a0 += p.x * v0; a1 += p.y * v0; a2 += p.z * v0; a3 += p.w * v0;
    b0 += p.x * v1; b1 += p.y * v1; b2 += p.z * v1; b3 += p.w * v1;
  }
  float* pp = part + ((long long)c * 8 + kv * 4) * 256 + 2 * f2;
  pp[0] = a0; pp[1] = b0; pp[256] = a1; pp[257] = b1; pp[512] = a2; pp[513] = b2; pp[768] = a3; pp[769] = b3;
}

extern "C" __global__ void k_attn_comb(const float* __restrict__ part, const float* __restrict__ gatev, float* __restrict__ att,
                            int nch){
  /* grid NH, block 256: sum the chunk partials in order, times the sigmoid gate */
  const int hh = blockIdx.x, f = threadIdx.x;
  float acc = part[hh * 256 + f];
  for(int c = 1; c < nch; ++c) acc += part[((long long)c * 8 + hh) * 256 + f];
  const float g = gatev[hh * 512 + 256 + f];
  att[hh * 256 + f] = acc * (1.0f / (1.0f + expf(-g)));
}


/* ---------------------------------------------------------------- head, argmax, sampling */

extern "C" __global__ void k_argmax(const float* __restrict__ y,
                         int n, int* __restrict__ out){
  __shared__ int bidx[1024];
  __shared__ float bval[1024];
  const int tid = threadIdx.x;
  float bv = -1e30f;
  int bi = 0;
  for(int e = tid; e < n; e += 1024){
    const float v = y[e];
    if(v > bv){ bv = v; bi = e; }
  }
  bidx[tid] = bi; bval[tid] = bv;
  __syncthreads();
  for(int st = 512; st > 0; st >>= 1){
    if(tid < st){
      const int i2 = tid + st;
      if(bval[i2] > bval[tid] ||
         (bval[i2] == bval[tid] && bidx[i2] < bidx[tid])){
        bval[tid] = bval[i2]; bidx[tid] = bidx[i2];
      }
    }
    __syncthreads();
  }
  if(tid == 0) *out = bidx[0];
}

extern "C" __global__ void k_expand(const int* __restrict__ sel, const int* __restrict__ gstart,
                                    const int* __restrict__ gcount, const int* __restrict__ off, int* __restrict__ glist){
  const int m = blockIdx.x, k = sel[m], n = gcount[k], o = off[m], s = gstart[k];
  for(int i = threadIdx.x; i < n; i += blockDim.x) glist[o + i] = s + i;
}

extern "C" __global__ void k_mask_neg(float* __restrict__ y, const int* __restrict__ tok, int n){
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    if(tok[i] < 0) y[i] = -1e30f;
}

extern "C" __global__ void k_band(const float* __restrict__ y, const int* __restrict__ amax, int n, float band,
                                  int* __restrict__ out_i, float* __restrict__ out_v, int* __restrict__ cnt, int cap){
  const float thr = y[amax[0]] - band;
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x){
    const float v = y[i];
    if(v >= thr){
      const int k = atomicAdd(cnt, 1);
      if(k < cap){ out_i[k] = i; out_v[k] = v; }
    }
  }
}
/* top-k version (one block of 1024): binary-search the largest threshold t in [max - band, max] with
   count(y >= t) >= k (16 halvings), then collect y >= t - so the candidate set always holds the true top-k */
extern "C" __global__ void k_topk(const float* __restrict__ y, const int* __restrict__ amax, int n, int k, float band,
                                  int* __restrict__ out_i, float* __restrict__ out_v, int* __restrict__ cnt, int cap){
  __shared__ int red[32];
  __shared__ float lo_s, hi_s;
  const int t = threadIdx.x, lane = t & 31, w = t >> 5;
  if(t == 0){ hi_s = y[amax[0]]; lo_s = hi_s - band; }
  __syncthreads();
  for(int it = 0; it < 17; ++it){
    const float mid = (it == 0) ? lo_s : 0.5f * (lo_s + hi_s);
    int c = 0;
    for(int i = t; i < n; i += 1024) c += (y[i] >= mid);
    for(int o = 16; o > 0; o >>= 1) c += __shfl_xor(c, o);
    if(lane == 0) red[w] = c;
    __syncthreads();
    if(t == 0){
      int tot = 0;
      for(int j = 0; j < 32; ++j) tot += red[j];
      if(it == 0){ if(tot < k) hi_s = lo_s; }   /* fewer than k within the band: keep the whole band */
      else if(tot >= k) lo_s = mid; else hi_s = mid;
    }
    __syncthreads();
    if(hi_s <= lo_s) break;
  }
  const float thr = lo_s;
  for(int i = t; i < n; i += 1024){
    const float v = y[i];
    if(v >= thr){
      const int q = atomicAdd(cnt, 1);
      if(q < cap){ out_i[q] = i; out_v[q] = v; }
    }
  }
}
