# FDDF
FDDF: Frequency Decomposition and Spatial-Frequency Dual-Domain Fusion Network for Multi-Spectral Pedestrian Detection

# FDDF: Frequency Decomposition and Spatial–Frequency Dual-Domain Fusion Network

This repository contains a reference implementation of the core modules of **FDDF** (Frequency Decomposition and Spatial–Frequency Dual-Domain Fusion Network) for multispectral pedestrian detection. The method is described in our paper:

> X. Liu, G. Xie, X. Xie, and X. Xu,  
> **"FDDF: Frequency Decomposition and Spatial-Frequency Dual-Domain Fusion Network for Multi-Spectral Pedestrian Detection"**,  


The full training and evaluation code (including data processing, training scripts, and model zoo) will be released **after the paper is officially accepted**. This repository currently focuses on the **two key building blocks** of our dual-domain fusion paradigm:

- **FDFD**: Frequency-Domain Feature Decomposition Module  
- **FSC**: Frequency–Spatial Domain Feature Global Co-occurrence Module
- (FSA and SDCI are conceptually included in the paper and will be released together with the complete code.)



Baseline Code and External References

Our implementation is built on top of existing open-source multispectral pedestrian detection codebases. In particular:

The training and detection pipeline (VGG-16 backbone + SSD detector, data loading, augmentation, and loss functions) is adapted from the official implementation of MLPD (“Multi-Label Pedestrian Detector in Multispectral Domain”) [Kim et al., RA-L 2021].https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection.git

Some utility functions (e.g., anchor generation, evaluation scripts) follow the design of standard SSD implementations in PyTorch.

dummy code1 and dummy code2 are reference codes for FDFD and FSG modules (for reference only)
Evaluation_stcript.cpy is runnable evaluation code
FDDF_desult. txt is the result of this article on Kaist
KAIST-annotation.json is a validation set label
KASIT_SENCHMARK.jpg is a comparison chart with other methods
