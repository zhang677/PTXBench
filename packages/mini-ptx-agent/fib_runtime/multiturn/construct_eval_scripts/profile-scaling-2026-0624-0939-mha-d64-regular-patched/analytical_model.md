# Analytical Model of First-Pass Trajectory Throughput

## Definitions

Let:

- \(w\) be the maximum number of concurrent trajectories (`MAX_PARALLEL`).
- \(N\) be the number of unique trajectories in the first pass. Here, \(N=80\).
- \(D\) be the maximum aggregate decode throughput, measured in trajectories/s.
- \(P\) be the maximum aggregate profiling throughput, measured in kernels/s.
- \(k\) be the number of kernels profiled per trajectory. Here, \(k=8\).
- \(\eta_D(w)\in[0,1]\) be decode-resource utilization at concurrency \(w\).
- \(\eta_P(w)\in[0,1]\) be profiling-resource utilization at concurrency \(w\).

The utilization terms account for the fact that low concurrency may not keep a
resource continuously busy.

## Decode and profiling time

At concurrency \(w\), the time needed to process the decode work is

\[
T_{\mathrm{decode}}(w)
=
\frac{N}{D\eta_D(w)}.
\]

Each trajectory produces eight profiling operations, so the profiling stage
must process \(8N\) kernels. Its required time is

\[
T_{\mathrm{profile}}(w)
=
\frac{8N}{P\eta_P(w)}.
\]

## Pipelined first-pass throughput

Decode and profiling work from different trajectories can overlap. In
steady-state operation, the slower pipeline stage therefore determines the
first-pass runtime:

\[
T_{\mathrm{first}}(w)
\approx
\max\left(
    \frac{N}{D\eta_D(w)},
    \frac{8N}{P\eta_P(w)}
\right).
\]

First-pass trajectory throughput is

\[
X_{\mathrm{first}}(w)
=
\frac{N}{T_{\mathrm{first}}(w)}.
\]

Substituting the runtime model gives the principal result:

\[
\boxed{
X_{\mathrm{first}}(w)
\approx
\min\left(
    D\eta_D(w),
    \frac{P\eta_P(w)}{8}
\right)
}
\]

with units of trajectories/s.

Thus, decode contributes a capacity of \(D\) trajectories/s, while profiling
contributes a capacity of only \(P/8\) trajectories/s because every trajectory
requires eight profiling operations.

## Explicit concurrency saturation model

A simple model for utilization increasing with concurrency is

\[
\eta_D(w)=1-e^{-w/w_D},
\qquad
\eta_P(w)=1-e^{-w/w_P},
\]

where \(w_D\) and \(w_P\) are characteristic concurrency levels needed to
saturate the decode and profiling resources.

The resulting throughput curve is

\[
\boxed{
X_{\mathrm{first}}(w)
\approx
\min\left[
    D\left(1-e^{-w/w_D}\right),
    \frac{P}{8}\left(1-e^{-w/w_P}\right)
\right].
}
\]

At low \(w\), increasing concurrency raises resource utilization. At high
\(w\), throughput approaches the pipeline capacity

\[
X_{\max}=\min\left(D,\frac{P}{8}\right).
\]

Additional workers cannot improve throughput after this bottleneck is
saturated. Queueing, scheduling overhead, or resource contention can cause a
small decline beyond the saturation point; that effect is not represented by
the basic saturation equation.

## Interpretation of the measured first pass

The measured first-pass results are:

| \(w\) | First-pass throughput (trajectories/hour) |
|---:|---:|
| 16 | 9.417 |
| 48 | 10.886 |
| 80 | 10.836 |

The increase from \(w=16\) to \(w=48\) indicates that \(w=16\) does not fully
utilize the pipeline. The nearly identical results at \(w=48\) and \(w=80\)
indicate saturation at approximately

\[
X_{\max}
\approx
10.86\ \text{trajectories/hour}
\approx
0.00302\ \text{trajectories/s}.
\]

The small decrease from \(w=48\) to \(w=80\) is consistent with measurement
variation or additional contention after reaching the bottleneck. It does not
indicate additional useful capacity at \(w=80\).

If profiling is the active bottleneck, its implied effective throughput is

\[
P
\approx
8X_{\max}
\approx
0.0241\ \text{kernels/s}
\approx
86.9\ \text{kernels/hour}.
\]

This corresponds to one completed profiling operation approximately every

\[
\frac{1}{P}\approx 41.4\ \text{seconds}.
\]

If decoding is the active bottleneck instead, the implied decode capacity is

\[
D
\approx
X_{\max}
\approx
0.00302\ \text{trajectories/s},
\]

or approximately one aggregate trajectory completion every \(331\) seconds.
Measurements of \(D\) and \(P\) are needed to determine which side of the
minimum is the actual bottleneck.

## Non-overlapped comparison

If decode and profiling could not overlap, their service times would add:

\[
T_{\mathrm{serial}}
=
N\left(\frac{1}{D}+\frac{8}{P}\right).
\]

The corresponding serial throughput would be

\[
X_{\mathrm{serial}}
=
\left(
    \frac{1}{D}+\frac{8}{P}
\right)^{-1}.
\]

The pipelined bottleneck equation is the appropriate primary model for the
concurrent watcher because different trajectories can decode and profile at
the same time.
