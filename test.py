# # from ultralytics import YOLO
# # import cv2
# # import os

# # model = YOLO("Weights/best.pt")
# # test_dir = r"C:\Users\user\Desktop\BikeHelmetDetector\TestImages"

# # correct = 0
# # total = 0

# # for img_name in os.listdir(test_dir):
# #     img_path = os.path.join(test_dir, img_name)
# #     img = cv2.imread(img_path)

# # print(f"Detection Accuracy on Test Images: {accuracy:.2f}%")
# #     results = model(img)
# #     for result in results:
# #         if len(result.boxes.cls) > 0:
# #             correct += 1  # detected something
# #         total += 1

# # accuracy = (correct / total) * 100
# from ultralytics import YOLO

# # Load model
# model = YOLO("Weights/best.pt")

# # Validate on test data
# metrics = model.val(data="data.yaml")
# print(metrics)  # Dictionary with precision, recall, mAP50, mAP50-95
from ultralytics import YOLO

# Path to your trained model
model_path = r"Weights\best.pt"

# Path to your dataset config file (data.yaml)
data_yaml_path = r"data.yaml"

# Load the trained YOLO model
model = YOLO(model_path)

# Validate the model on the dataset
metrics = model.val(data=data_yaml_path, device="cpu")  # Change to "cuda" if you have GPU

# Print results
print("\n📊 Validation Results:")
print(f"Precision: {metrics.box.map:.4f}")   # Overall precision (mAP50)
print(f"mAP50:     {metrics.box.map50:.4f}") # mAP at IoU=0.5
print(f"mAP50-95:  {metrics.box.map95:.4f}") # mAP averaged over IoU thresholds

# Per-class metrics
if hasattr(metrics, "names"):
    print("\nPer-class Performance:")
    for i, class_name in metrics.names.items():
        print(f"  {class_name} -> mAP50: {metrics.box.maps[i]:.4f}")
