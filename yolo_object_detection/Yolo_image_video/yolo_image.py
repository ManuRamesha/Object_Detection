import cv2
import time
from ultralytics import YOLO

# Load the YOLO model
def load_model(model_path: str):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Get the class colors for bounding boxes
def get_class_colors(class_number: int) -> tuple:
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = class_number % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        base_colors[color_index][i] + increments[color_index][i] * (class_number // len(base_colors)) % 256
        for i in range(3)
    ]
    return tuple(color)

# Process the image for object detection
def process_image(image, model, class_names):
    results = model(image)  # Run object detection on the image
    for result in results:
        for box in result.boxes:
            # Check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # Extract coordinates and convert to integers
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class index and name
                class_index = int(box.cls[0])
                class_name = class_names[class_index]

                # Get bounding box color
                color = get_class_colors(class_index)

                # Draw bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f'{class_name} {box.conf[0]:.2f}', (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image

# Main function to run object detection on an image
def main():
    model_path = 'yolov8s.pt'

    # Load the YOLO model
    model = load_model(model_path)
    if model is None:
        return  # Exit if model loading fails

    # Load the image for object detection
    image_path = 'input_image.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image.")
        return

    # Retrieve class names from the model
    class_names = model.names

    # Process the image
    image = process_image(image, model, class_names)

    # Display the image with detected objects
    cv2.imshow('Object Detection', image)

    # Optionally save the output image
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to {output_path}")

    # Wait for user input to close (with timeout to ensure exit)
    start_time = time.time()
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # If 'q' is pressed, exit
            break
        if time.time() - start_time > 5:  # Timeout after 5 seconds
            print("Exiting after timeout.")
            break

    cv2.destroyAllWindows()  # Close all windows
    print("Exiting...")

if __name__ == '__main__':
    main()
