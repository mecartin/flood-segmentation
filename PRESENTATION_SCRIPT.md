# 🎙️ Weather-Aware Flood Segmentation: 5-Minute Presentation Guide

This document contains a structured speech and instructions for presenting your Multimodal Weather-Aware U-Net project in a 5-minute window. 

---

## ⏱️ Logistics & Demo Instructions

**Before you start speaking:**
1. Open up the newly completely `demo_comparison.png` in an image viewer, or keep the terminal open.
2. If you are doing a **live code execution**, cue up your terminal so you can simply press <kbd>Enter</kbd> on the following command when you reach "Section 5":
   ```bash
   python scripts/demo.py --samples 4
   ```

**Tone:** Academic, investigative, and transparent. The most compelling part of this project isn't just the architecture, but *the critical analysis of why multimodal fusion failed to scale* due to overfitting.

---

## 🗣️ The Script

### 1. The Hook & Introduction (1:00 min)
**[Slide/Screen: Show the title screen or a raw Sentinel-1 satellite image containing a flood]**

"Hi everyone. Today I’ll be walking you through my research on multimodal deep learning for disaster response—specifically, mapping floods from space. 

When emergency responders need to locate floods, they rely on Synthetic Aperture Radar (SAR) imagery, like Sentinel-1 data, because it can see through clouds and darkness. However, radar is notoriously noisy. It bounces off buildings and vegetation in chaotic ways, making standing water very hard to isolate. 

The core research question of this project was: **Can we make a neural network smarter at flood detection by giving it real-time meteorological context?** If the network *knows* the temperature, humidity, and recent precipitation, can it better distinguish an actual flood from a dark asphalt parking lot? To test this, I built a 'Weather-Aware' Multimodal U-Net and compared it directly against a standard baseline."

### 2. The Architecture (1:00 min)
**[Slide/Screen: Show your README architecture diagrams or Project Structure, or `02_baseline_unet.ipynb` / `03_multimodal_unet.ipynb`]**

"My pipeline started with the Sen1Floods11 dataset. For my **Baseline Model** (Model A), I built a robust U-Net equipped with 'CBAM' Attention blocks, forcing the network to dynamically spotlight only the most relevant spatial and channel features. 

For the **Multimodal Model** (Model B), things got more complex. You can't just concatenate a 5-dimensional weather vector onto an image. Instead, I used a technique called **FiLM**—Feature-wise Linear Modulation. A multi-layer perceptron analyzes the weather data and generates shift and scale modifiers. I injected these into *all four levels* of the U-Net's decoder. Essentially, the weather data acts as a dynamic 'gate', physically modulating how the network interprets visual satellite features based on the climate conditions."

### 3. Engineering Challenges & Solutions (1:00 min)
**[Slide/Screen: Show some code from `src/losses.py` or highlight the `PROJECT_PROGRESS.md` findings]**

"Training this was an immense engineering challenge. 
First, floods are rare. The vast majority of a satellite image is dry land. Standard losses failed. I implemented a heavy custom loss function called **Tversky Focal Loss** to harshly penalise the network for missing rare flood pixels, forcing it to tackle the class imbalance.

Second, I encountered a silent PyTorch mathematical freeze. When summing spatial gradients in float16 mixed-precision, the loss was overflowing the maximum limits of FP16 memory. Force-casting to FP32 before the summation revived the pipeline."

### 4. Results & Discoveries (1:00 min)
**[Slide/Screen: Show the `04_ablation_barchart.png` or `04_iou_distributions.png`]**

"The results were deeply interesting, but counter-intuitive.
The Baseline Model hit a respectable Test IoU of roughly 0.466. 
However, the Multimodal Weather-Aware model **degraded** performance across most metrics, dropping the IoU to 0.434. 

Why? Because the weather gating was *too* powerful. It caused severe overfitting. The network stopped learning safe visual geographic features and instead 'cheated' by latching onto spurious correlations in the weather data—becoming heavily reliant on the training set's climate rather than the satellite geometry. It gained higher precision, meaning fewer false alarms, but its recall plummeted as it became incredibly conservative."

### 5. Live Demo & Lessons Learned (1:00 min)
**[Slide/Screen: Run `python scripts/demo.py` and open `results/figures/demo_comparison.png`]**

"Here is a side-by-side demo of both models running inference on unseen Test data. *(Point to the grid)*
On the right, you can see how Model B alters its predictions compared to the Baseline, becoming more restricted.

The ultimate takeaway from this architecture is a lesson in **multimodal generalisation**. Moving forward, the fix for this is aggressive 'Modality Dropout'—shutting off the weather data for 30% of training batches, and skyrocketing internal Dropout to 0.5. By preventing the network from becoming codependent on its secondary modality, we force the vision weights to stay strong.

Thank you for listening—I'd be happy to take any questions."
