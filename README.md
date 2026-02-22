# FDDF

## Frequency Decomposition and Spatial--Frequency Dual-Domain Fusion Network for Multi-Spectral Pedestrian Detection

```{=html}
<p align="center">
```
`<img src="figures/fddf_framework.png" width="85%">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 Overview

This repository provides the reference implementation of **FDDF**
(Frequency Decomposition and Spatial--Frequency Dual-Domain Fusion
Network), a dual-domain fusion framework designed for multispectral
pedestrian detection.

FDDF explicitly models complementary information in:

-   **Frequency domain** (low-frequency structure + high-frequency
    details)
-   **Spatial domain** (local-global contextual co-occurrence)

Core modules:

-   **FDFD** -- Frequency-Domain Feature Decomposition\
-   **FSC** -- Frequency--Spatial Domain Global Co-occurrence Modeling

The complete training and inference pipeline will be released after
official paper acceptance.

------------------------------------------------------------------------

## 🧠 Method Architecture

FDDF follows a dual-domain fusion paradigm:

1.  RGB--Thermal feature extraction\
2.  Frequency decomposition (low/high separation)\
3.  Spatial-frequency global co-occurrence modeling\
4.  Adaptive feature fusion for detection

```{=html}
<p align="center">
```
`<img src="figures/fddf_pipeline.png" width="80%">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 📊 Experimental Results

### 🔹 Qualitative Visualization

```{=html}
<p align="center">
```
`<img src="/kaist-res4.pdf" width="85%">`{=html}
```{=html}
</p>
```
FDDF improves detection robustness under:

-   Low illumination
-   Occlusion
-   Thermal noise interference

------------------------------------------------------------------------

### 🔹 KAIST Benchmark Comparison

```{=html}
<p align="center">
```
`<img src="figures/kaist_benchmark.jpg" width="75%">`{=html}
```{=html}
</p>
```
FDDF achieves competitive performance on:

-   MR (Reasonable / All)
-   Day/Night subsets
-   Cross-modal robustness

------------------------------------------------------------------------

### 🔹 Training Convergence Curve

```{=html}
<p align="center">
```
`<img src="figures/training_curve.png" width="75%">`{=html}
```{=html}
</p>
```
Dual-domain modeling improves convergence stability and optimization
behavior.

------------------------------------------------------------------------


------------------------------------------------------------------------

## ⚙️ Baseline and References

Training pipeline adapted from:

MLPD -- Multi-Label Pedestrian Detection (RA-L 2021)\
https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection.git

------------------------------------------------------------------------

## 📄 Citation

If you find this work useful, please cite:

    @article{liu2026fddf,
      title={FDDF: Frequency Decomposition and Spatial–Frequency Dual-Domain Fusion Network for Multi-Spectral Pedestrian Detection},
      author={Liu, Xiaowei and Xie, Guang and Xie, Xiangyu and Xu, Xiaodong},
      journal={IEEE Transactions on ...},
      year={2026}
    }

------------------------------------------------------------------------

## 📜 License

Released for academic research only.

For commercial use, please contact the authors.

------------------------------------------------------------------------

*Generated on 2026-02-22*
