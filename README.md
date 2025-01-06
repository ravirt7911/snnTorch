# Spiking Neural Networks on the DVS Gesture 128 Dataset

This repository presents the implementation and evaluation of a Spiking Neural Network (SNN) on the DVS Gesture 128 dataset using the snnTorch model. The study highlights the biologically inspired approach of SNNs for event-driven data processing, leveraging surrogate gradient descent for training and achieving a validation accuracy of 87.50%.

---

## Abstract

Spiking Neural Networks (SNNs) utilize discrete spikes to emulate biological neural communication. This study implements an SNN trained on the DVS Gesture 128 dataset using the snnTorch library. The network employs a single-layer architecture with Leaky Integrate-and-Fire (LIF) neurons, achieving 88.28% accuracy. The work demonstrates the effectiveness of SNNs for event-driven tasks and underscores the utility of snnTorch for SNN research and experimentation.

---

## Data Set Link - 
[DataSet](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)

## Keywords

- Spiking Neural Networks  
- Surrogate Gradient Descent  
- snnTorch  
- Leaky Integrate-and-Fire Neurons  
- DVS Gesture 128 Dataset  
- Neuromorphic Computing  
- Event-Driven Data Processing  

---

## About the Dataset

The **DVS Gesture 128 dataset** is a neuromorphic dataset comprising event-based recordings of hand gestures. Key characteristics include:
- **Spatial and Temporal Resolution:** 128x128 pixels.
- **Event-Driven Data:** Captures asynchronous event streams over time.

This dataset is ideal for evaluating SNNs due to its inherent temporal dynamics and event-driven structure.

---

## Proposed Algorithm

### Preprocessing
1. Normalize event data for consistent scaling.
2. Encode input events into spike trains using rate-based encoding over a 100ms simulation window.

### Network Architecture
- **Input Layer:** Encodes spike trains from DVS Gesture data.
- **Spiking Layer:** Single-layer LIF neurons with:
  - Membrane potential dynamics.
  - Threshold-triggered spiking and reset mechanisms.
- **Output Layer:** Spike counts mapped to gesture class probabilities.

### Training Process
- **Loss Function:** Cross-entropy adapted for spike outputs.
- **Optimizer:** Adam optimizer with a learning rate of 0.001.
- **Training Epochs:** 10 epochs with a batch size of 128.
- **Surrogate Gradients:** Enabled backpropagation for spiking neurons.

---

## Results and Performance

### Classification Results
- **Final Validation Accuracy:** 87.50%.

### Visualizations
- **Training Curves:** Monitor loss convergence and accuracy trends.
- **Spike Raster Plots:** Display neuron firing activity over time.
- **Confusion Matrices:** Summarize gesture classification performance.

### Neuron Behavior
- LIF neurons exhibited selective firing patterns, indicating learned gesture representations.

---

## Reproducibility

This implementation leverages the snnTorch library, which extends PyTorch for SNN modeling. The code and methodology ensure reproducibility on hardware platforms supporting Python and GPU acceleration.

---

## Dependencies and Requirements

### Dependencies
- **Python Libraries:** `numpy`, `pandas`,`pytorch`, `matplotlib`, `seaborn`, `snntorch`, `scikit-learn`.

### Minimum Requirements
- **RAM:** 8GB.  
- **Processor:** Dual-Core 2.0 GHz.  

### Recommended Requirements
- **RAM:** 16GB.  
- **Processor:** Quad-Core 2.5 GHz or higher.  
- **GPU:** For faster computations.

### Installation
Install dependencies using pip:
```bash
pip install numpy pandas matplotlib seaborn snntorch scikit-learn



