## My Crowd Experiment

**Implementation Notes:**
- The loss function must be properly **normalized** to ensure stable training; however, **Sigmoid should not be applied indiscriminately**, as it may distort regression-based objectives.
- Input data follow the conversion pipeline:  
  `PIL.Image → numpy.asarray → torch.Tensor`.
- **Data Shape Transformation:**  
  `Original image (PIL / OpenCV): (H, W) / (H, W, C) -> NumPy / PIL(RGB): (H, W, 3) -> Tensor: (3, H, W) -> DataLoader batch: (B, C, H, W)`
- **Image Classification:**  
  `(B, C, H, W) -> (B, C) -> (B, num_classes)`
- **Semantic Segmentation:**  
  `(B, C, H, W) -> (B, num_classes, H, W)`
- **Crowd Counting:**  
  `(B, 1, H, W) -> List[points] per image`
- **Visualization:**  
  `(B, C, H, W) -> (C, H, W) -> (H, W, C) -> NumPy / PIL`
- During validation, testing, and visualization, **foreground** and **background** regions must be explicitly distinguished.
- Configuration management should be handled via `configs.yaml` to ensure reproducibility and clarity.
- The **LMDS** module outputs **point count, coordinates, labels, and confidence scores**, all in **list format**.
- **PointToMask** generates the ground-truth **hard disk mask** from point-level annotations.
- **FIDT** and **PointToMask** accept inputs in either `PIL.Image` or `torch.Tensor` format.  
  - **FIDT** expects input with shape `(C, H, W)`.  
  - **PointToMask** supports inputs of shape `(H, W)` or `(C, H, W)`.
<p align="center">
  <img src="datasets/fidt.png" width="700">
</p>
<p align="center">
  <img src="datasets/ptm.png" width="700">
</p>