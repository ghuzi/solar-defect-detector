# Solar Panel Defect Detector

A YOLOv8 web app that classifies the condition of a solar panel photo as **clean**, **dust**, **bird_drops** or **cracked**, so maintenance teams know which panels need cleaning and which need repair or replacement.

**Live demo:** https://solar-defect-detector.streamlit.app/

## Business problem
Dust, bird droppings and cracks reduce the output of solar installations. Manual inspection is slow and expensive. This tool gives a fast first check from a single photo, with a suggested action for each finding.

## Model
- Base model: `yolov8n.pt` (pretrained on COCO), fine-tuned for 25 epochs at 640x640 with Ultralytics
- Dataset: [solar panal](https://universe.roboflow.com/polytechnic-university-ubjwp/solar-panal) by polytechnic university on Roboflow Universe (CC BY 4.0), version 1: 6,552 images (train 5,662 / valid 594 / test 296), 4 classes

## Results (held-out test set, 296 images)
| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| all | 0.775 | 0.631 | 0.699 |
| clean | 0.912 | 0.783 | 0.846 |
| dust | 0.728 | 0.793 | 0.841 |
| cracked | 0.842 | 0.713 | 0.773 |
| bird_drops | 0.620 | 0.237 | 0.336 |

## Limitations
- `bird_drops` is the weakest class: droppings are small and the model misses many of them at the default threshold. Lowering the confidence slider helps.
- The dataset labels whole panels, so the boxes describe the panel's condition rather than the exact pixel location of a defect.

## App features
- Upload a JPG/PNG image
- Confidence threshold slider
- Image with bounding boxes and labels
- Table of detected objects with confidence scores and a suggested action

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py`: Streamlit interface
- `models/best.pt`: trained weights
- `.streamlit/config.toml`: theme
