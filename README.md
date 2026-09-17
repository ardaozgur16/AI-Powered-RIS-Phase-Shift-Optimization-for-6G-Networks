# AI-Powered RIS Phase Shift Optimization for 6G Networks

> 🎓 **Acknowledgement:**
> This project was developed alongside insights gained from the **University of Glasgow - 6G Vision: ML, Intelligent Surfaces & Optical Networks** program.
> 
> 
> **Credential ID:** [D4NFS910CO9Y](https://www.google.com/search?q=https://coursera.org/verify/D4NFS910CO9Y)[cite: 7]

---

## 🚀 Key Highlights

* **Realistic Channel Modeling:** 3D coordinate-based topology incorporating cascaded geometric path loss, Rician (LoS) fading for RIS paths, and Rayleigh (NLoS) fading for direct paths[cite: 5, 8].
* **Unsupervised Policy Network:** Deep neural architecture trained to maximize spectral efficiency without labeled optimal phases, achieving sub-millisecond inference[cite: 3, 8].
* **Rigorous Benchmarks:** Comparative performance analysis against No-RIS, Random Phase, and Alternating Optimization (AO) theoretical upper bounds[cite: 1, 8].
* **Hardware Feasibility:** Discrete 1-bit and 2-bit phase-shift quantization analysis modeling practical metasurface constraints[cite: 1, 8].
* **High-Efficiency Scaling:** Delivers up to a **591x latency speedup** over iterative matrix solvers as array size scales to $N=128$ elements[cite: 1].

---

## 📂 Project Structure

* `channel_model.py`: 3D wireless channel realization, path loss computations, and small-scale fading generators[cite: 5, 8].
* `baselines.py`: Benchmark solvers including direct path (No-RIS), random phase perturbation, and iterative Alternating Optimization (AO)[cite: 4, 8].
* `environment.py`: Gymnasium-compatible MISO-RIS wireless simulation environment[cite: 2, 8].
* `train.py`: PyTorch-based training pipeline and model persistence (`ris_model.pth`)[cite: 3, 8].
* `plots_and_benchmarks.py`: Transmit Power vs. Spectral Efficiency evaluation and CPU/GPU inference latency analysis[cite: 1, 8].

---

## 🛠️ Setup & Execution

### Prerequisites

```bash
pip install -r requirements.txt

```

### 1. Train the Optimization Network

Trains the unsupervised RIS phase-prediction network and saves the model weights[cite: 3, 8]:

```bash
python train.py

```

### 2. Run Benchmarks & Generate Plots

Executes Monte Carlo trials across varying transmit powers and profiles per-channel latency[cite: 1, 8]:

```bash
python plots_and_benchmarks.py

```

---

## 📊 Key Results

| Metric | Alternating Optimization (AO) | AI-Powered NN (CPU) | AI-Powered NN (GPU) |
| --- | --- | --- | --- |
| **Spectral Efficiency ($P_t=30\text{ dBm}$)** | $\sim 10.9\text{ bps/Hz}$ | $\sim 10.8\text{ bps/Hz}$ | $\sim 10.8\text{ bps/Hz}$ |
| **Inference Time ($M=4, N=16$)**[cite: 1] | $28.45\text{ ms}$[cite: 1] | $0.32\text{ ms}$ (~89x speedup)[cite: 1] | $0.08\text{ ms}$ (~350x speedup)[cite: 1] |
| **Latency Feasibility** | Exceeds coherence time | Real-time URLLC viable | Real-time URLLC viable |
