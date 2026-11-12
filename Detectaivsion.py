import cv2 
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

cap = cv2.VideoCapture(0)

print("Open camera... ")
print("Press q to close")

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.5, verbose=False)

    for box in results[0].boxes:

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

        info = f"{label} (conf:.2f) - Tam:({cx},{cy})"
        cv2.putText(frame, info, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("AI vision - Ex2", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()