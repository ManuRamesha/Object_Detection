import cv2
from ultralytics import YOLO

# Load the YOLO model
def load_model(model_path: str):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Get the class colors from bounding boxes
def get_class_colors(class_number: int) -> tuple:
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = class_number % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * (class_number // len(base_colors)) % 256
             for i in range(3)]
    return tuple(color)

# Process each frame for object detection
def process_frame(frame, model, class_names):
    results = model(frame)  # Detect objects in the frame
    for result in results:
        for box in result.boxes:
            # Check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # Extract coordinates, convert to integers
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class index and name
                class_index = int(box.cls[0])
                class_name = class_names[class_index]

                # Get bounding box color
                color = get_class_colors(class_index)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

# Main function
def main():
    model_path = 'yolov8s.pt'

    # Load the YOLO model
    model = load_model(model_path)
    if model is None:
        return  # Exit if model loading fails

    # Path to your video file
    video_path = 'video.mp4'

    # Open video capture
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Error: Could not open video capture.")
        print(f"Video file path: {video_path}")
        return

    # Retrieve class names from the model
    class_names = model.names

    # Desired width and height for resizing
    new_width = 1920
    new_height = 1080

    while True:
        ret, frame = video_cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process the frame for object detection
        frame = process_frame(frame, model, class_names)

        # Resize the frame to the desired size
        frame_resized = cv2.resize(frame, (new_width, new_height))

        # Show the processed frame
        cv2.imshow('YOLO Object Detection', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
