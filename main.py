import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# open video
# model.track(source="football.mp4", show=True)



# open image
image_path = 'tennis.png'
image = cv2.imread(image_path)

results = model.track(image, persist=True)
annotated_image = results[0].plot()

cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# train with own dataset
# model.train(data="data.yaml", epochs=2, imgsz=640)