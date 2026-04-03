# Project Progress: Flood Segmentation

## Status: Baseline vs. Multimodal Evaluation

**Date:** April 2026
**Summary:** Evaluated the Baseline U-Net against the Multimodal model (which incorporates additional features).

### Findings from Ablation Summary
- **Baseline Model:** IoU ~0.466, Dice ~0.612, Loss ~0.332
- **Multimodal Model:** IoU ~0.434, Dice ~0.574, Loss ~0.369
- **Comparison:** The Multimodal model degraded performance across most metrics (IoU drop of 0.032, Dice drop of 0.037) compared to the Baseline. 
- **Behavior Shift:** The Multimodal model showed higher Precision (+0.034) but significantly lower Recall (-0.057). It became more conservative, predicting fewer positive (flood) pixels, which reduced false positives but heavily increased false negatives.

### Identified Issues
1. **Multimodal Fusion Strategy:** Simply concatenating or naively fusing features is likely causing the network to ignore the satellite imagery or the features act as noise.
2. **Class Imbalance:** Lower recall indicates trouble identifying positive flood pixels, which is common in highly imbalanced datasets.

### Next Steps for Improvement (Completed)
- **Implemented `FocalDiceLoss`:** Replaced the simple BCE-Dice loss. Focal loss dynamically scales cross entropy based on prediction confidence, forcing the model to focus on the hard positive (flood) pixels instead of easy background pixels.
- **Upgraded Architecture (FiLM):** Replaced the naive concatenation in the multimodal U-Net's bottleneck with Feature-wise Linear Modulation (FiLM). The weather embedding now generates shift (`beta`) and scale (`gamma`) parameters that directly modulate the spatial feature maps. This allows weather to dynamically "gate" which visual features the network pays attention to.
