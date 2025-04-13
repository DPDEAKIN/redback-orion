import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import json
from datetime import datetime
from kafka import KafkaProducer

# === Kafka Producer Setup ===
# Ensure that Kafka is running and accessible at the specified address.
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda m: json.dumps(m).encode('utf-8')
)
kafka_topic = 'detections_topic'

# === YOLO Model Setup ===
# Paths to your YOLO weights, config, and COCO labels.
yolo_weights = 'yolov4.weights'
yolo_config = 'yolov4.cfg'
coco_names = 'coco.names'

# Load the YOLO network.
net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
layer_names = net.getLayerNames()
# Get the names of the output layers.
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open(coco_names, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# === Heatmap Initialise ===
# Define the dimensions for the heatmap grid.
heatmap_size = (100, 100)
heatmap = np.zeros(heatmap_size)

# === Video Stream Setup ===
# For a live feed, change 'video_source' to 0 (webcam), an RTSP URL, etc.
video_source = 0  # Replace with your live feed source as needed.
cap = cv2.VideoCapture(video_source)

# === Processing Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    
    # Prepare the frame for YOLO by creating a blob.
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Process YOLO outputs to detect persons.
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes[class_id] == 'person' and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                
                # Map the centre coordinates to the heatmap grid.
                x = int((center_x / width) * heatmap_size[1])
                y = int((center_y / height) * heatmap_size[0])
                heatmap[y, x] += 1

                # === Publish Detection Event to Kafka ===
                detection_event = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'class': 'person',
                    'confidence': float(confidence),
                    'center_x': center_x,
                    'center_y': center_y,
                    'heatmap_grid_x': x,
                    'heatmap_grid_y': y
                }
                producer.send(kafka_topic, detection_event)
    
    # Create a heatmap overlay on the live video frame.
    heatmap_vis = cv2.resize(heatmap, (width, height))
    # Normalise only if there is at least one detection.
    if heatmap_vis.max() > 0:
        normalised_heatmap = (heatmap_vis / heatmap_vis.max() * 255).astype(np.uint8)
    else:
        normalised_heatmap = heatmap_vis.astype(np.uint8)
    heatmap_colour = cv2.applyColorMap(normalised_heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_colour, 0.4, 0)
    


    cv2.imshow('Live Heatmap Overlay', overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === 2D Heatmap Visualisation ===
plt.figure(figsize=(8, 6))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.title('2D Crowd Heatmap')
plt.xlabel('Grid X')
plt.ylabel('Grid Y')
plt.colorbar(label='Detection Count')
plt.show()

# === 3D Heatmap Visualisation ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X = np.arange(0, heatmap_size[1])
Y = np.arange(0, heatmap_size[0])
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, heatmap, cmap='hot')
ax.set_title('3D Crowd Heatmap')
ax.set_xlabel('Grid X')
ax.set_ylabel('Grid Y')
ax.set_zlabel('Detection Count')
plt.show()
