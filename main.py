# Parker Wilson
import cv2
from ultralytics import YOLO
from time import perf_counter
from tracking import calculate_direction

def main():

    try:
        target = {"x": None, "y": None} 

        camera_index = 0 
        model_path = "yolo26n.pt"
        confidence_threshold = 0.5
        image_size = 320
        dead_zone = 20
        alpha = 0.3
        camera = cv2.VideoCapture(camera_index)

        if camera.isOpened():
            print("Camera opened successfully.")

            # Create the window before attaching its mouse callback.
            cv2.namedWindow("Person Detection Software")

            detector = YOLO(model_path)

            smoothed_x = None
            smoothed_y = None

            tracking_enabled = True
            while True:
                frame_start = perf_counter()
                success, frame = camera.read()

                # Check the frame before trying to use it.
                if not success:
                    print("Failed to read frame from camera.")
                    break

                results = detector.predict(
                    frame,
                    classes=[0],
                    conf=confidence_threshold,
                    verbose=False,
                    imgsz=image_size
                )


                # Get the detected people's box coordinates.
                person_boxes = results[0].boxes.xyxy

                target["x"] = None
                target["y"] = None

                # Use the first person's center as our target.
                if len(person_boxes) > 0:
                    left, top, right, bottom = person_boxes[0].tolist()
                    detected_x = (left + right) / 2
                    detected_y = (top + bottom) / 2

                    if smoothed_x is None:
                        smoothed_x = detected_x
                        smoothed_y = detected_y
                    else:
                        smoothed_x += alpha * (detected_x - smoothed_x)
                        smoothed_y += alpha * (detected_y - smoothed_y)
                    target["x"] = int(smoothed_x)
                    target["y"] = int(smoothed_y)
                else:
                    smoothed_x = None
                    smoothed_y = None

                frame = results[0].plot()

                height, width = frame.shape[:2]
                center_x = width // 2
                center_y = height // 2

                overlay = frame.copy()
                
                # 2. Draw a filled red circle corresponding to the dead_zone radius on the overlay
                cv2.circle(
                    overlay, 
                    (center_x, center_y), 
                    dead_zone, 
                    (0, 0, 255), 
                    -1
                )
                
                # 3. Blend the overlay back into the main frame (0.25 opacity for the red circle)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            
                horizontal = "STOP"
                vertical = "STOP"
                status = "NO PERSON"

                if not tracking_enabled:
                    status = "PAUSED"
                if tracking_enabled and target["x"] is not None and target["y"] is not None:
                    status = "TRACKING"
                    cv2.circle(
                        frame,
                        (target["x"], target["y"]),
                        5,
                        (0, 0, 255),
                        -1
                    )
                    error_x = target["x"] - center_x
                    error_y = target["y"] - center_y

                    horizontal = calculate_direction(error_x, dead_zone)
                    vertical = calculate_direction(error_y, dead_zone)

                    cv2.putText(
                        frame,
                        f"Error: x={error_x}, y={error_y}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                print(f"X Error: {error_x} | Y Error: {error_y}")

                cv2.putText(
                    frame,
                    f"{status} | Pan: {horizontal}, Tilt={vertical}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                elapsed = perf_counter() - frame_start
                fps = 1 / elapsed

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Person Detection Software", frame)

                # Process window events and check for a keypress.
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty("Person Detection Software", cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                elif key == ord("q"):
                    break
                elif key == ord (" "):
                    tracking_enabled = not tracking_enabled

        else:
            print("Failed to open camera.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
