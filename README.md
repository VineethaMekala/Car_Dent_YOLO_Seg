# Car_Dent_YOLO_Seg
# Vehicle Dent Detection using YOLOv8 Segmentation

This model implements an **AI-powered vehicle dent detection system** using **instance segmentation**. The model identifies dents on vehicle surfaces and highlights the exact damaged regions using pixel-level masks.

The system is built using the YOLOv8 Segmentation framework from **Ultralytics**, which enables real-time object detection and segmentation.

## Project Objective

Traditional object detection only draws bounding boxes. However, vehicle damage analysis requires **precise shape detection** of dents. This project solves that by:

* Detecting dent regions
* Segmenting the exact dent shape
* Measuring affected area using pixel masks

## Model Overview

The system uses a YOLOv8 segmentation model pre-trained on general datasets and fine-tuned on a custom dent dataset.

The model performs:

* Object detection
* Instance segmentation
* Confidence scoring

---

## Dataset Structure

The dataset follows the YOLO segmentation format:

```
dent_dataset/
│
├── train/
│    ├── images/
│    └── labels/
│
├── val/
│    ├── images/
│    └── labels/
```

Each image has a matching label file containing polygon coordinates of dent regions.

## Annotation Format

For every dent, the annotation includes:

* Class ID
* Polygon points outlining dent boundary

Coordinates are normalized between 0 and 1.

## Training Process

The YOLOv8 segmentation model is trained using:

* Input image size of 640×640
* Single-class dent detection
* Multiple training epochs for convergence

The training pipeline allows the model to learn dent shapes, sizes, and visual patterns from the dataset.

## Inference Process

During testing, the trained model:

1. Receives a vehicle image
2. Detects dent regions
3. Generates a segmentation mask
4. Overlays a transparent red mask on the damaged area
5. Draws bounding boxes
6. Outputs confidence and severity metrics

The result is an easy-to-interpret visual representation of vehicle damage.

## Output

The system produces:

* Transparent segmentation masks
* Bounding boxes
* Severity estimation (based on mask area)
* Confidence score

## Applications

This system can be used in:

* Automotive inspection systems
* Insurance claim assessment
* Manufacturing defect detection
* Smart garage automation

## Performance Considerations

* Lightweight model for real-time inference
* Works on GPU and CPU
* Suitable for edge deployment

## Acknowledgment

This project uses the YOLOv8 framework developed by **Ultralytics** for object detection and segmentation.
