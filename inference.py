from ultralytics import YOLO
import cv2
import numpy as np

# ----------------------------------
# 1️⃣ Load trained model
# ----------------------------------
model = YOLO("best.pt")  # path to your trained model

# ----------------------------------
# 2️⃣ Run prediction
# ----------------------------------
results = model.predict(r"C:\Users\valkontek 032\Desktop\Practice Program\YOLO_Seg_model\dent_seg_dataset\Valid\images\70.jpeg", conf=0.4, imgsz=640)

# ----------------------------------
# 3️⃣ Process result
# ----------------------------------
for r in results:
    img = r.orig_img.copy()

    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        overlay = img.copy()

        for i, mask in enumerate(masks):
            mask = mask.astype(np.uint8)

            # 🔥 Resize mask to original image size
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            red_mask = np.zeros_like(img)
            red_mask[:, :, 2] = mask * 255

            overlay = cv2.addWeighted(overlay, 1.0, red_mask, 0.5, 0)

            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)


        # ----------------------------------
        # 4️⃣ Save output
        # ----------------------------------
        cv2.imwrite("yolo_seg_output.jpg", overlay)
        print("🖼 Saved: yolo_seg_output.jpg")
    else:
        print("No objects detected.")
