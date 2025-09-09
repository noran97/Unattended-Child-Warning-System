import cv2
import numpy as np
import tensorflow as tf
import argparse
import os
import time
import requests
import json
from datetime import datetime
import uuid
from urllib.parse import urlparse, urlunparse


class AzureNotificationSender:
    def __init__(self, azure_url: str):
        self.azure_url = azure_url
        # Assuming you'll have some headers, e.g., for content type
        self.headers = {'Content-Type': 'application/json', 'x-ms-blob-type': 'BlockBlob'}


    def _build_blob_url(self, filename: str) -> str:
        parsed = urlparse(self.azure_url)
        container_url = urlunparse(parsed._replace(query=''))
        return f"{container_url.rstrip('/')}/{filename}?{parsed.query}"

    def send_notification(self, message, alert_type="unattended_children", elapsed_time=0):
        try: # Add try block here
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            notification_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "alert_type": alert_type,
                "message": message,
                "elapsed_time_seconds": elapsed_time,
                "notification_id": str(uuid.uuid4())
            }

            filename = f"alert_{timestamp}_{notification_data['notification_id'][:8]}.json"
            notification_url = self._build_blob_url(filename)

            response = requests.put(
                notification_url,
                data=json.dumps(notification_data, indent=2).encode("utf-8"),
                headers=self.headers,
                timeout=10
            )
            if response.status_code in [200, 201]:
                    print(f"Notification sent successfully to Azure: {filename}")
                    print(f"Alert ID: {notification_data['notification_id']}")
                    return True
            else:
                    print(f"❌ Failed to send notification. Status: {response.status_code}")
                    print(f"Response: {response.text}")
                    return False

        except requests.exceptions.Timeout:
            print("⏱️ Azure notification timeout - network may be slow")
            return False
        except requests.exceptions.ConnectionError:
            print("🌐 Azure notification failed - check internet connection")
            return False
        except Exception as e:
            print(f"❌ Azure notification error: {str(e)}")
            return False


class UnattendedChildrenTracker:
    def __init__(self, warning_threshold=60, azure_url=None):
        """
        Initialize unattended children tracker with Azure notifications

        Args:
            warning_threshold (int): Time in seconds before triggering warning
            azure_url (str): Azure Blob Storage URL for notifications
        """
        self.warning_threshold = warning_threshold
        self.children_only_start_time = None
        self.last_notification_time = None
        self.notification_cooldown = 300  # 5 minutes cooldown between notifications
        self.warning_active = False

        # Initialize Azure notification sender
        self.azure_sender = AzureNotificationSender(azure_url) if azure_url else None


        self.last_escalation_level = -1

    def update_detection(self, has_children, has_adults):
        """
        Update detection state and handle notifications

        Args:
            has_children (bool): Whether children are detected
            has_adults (bool): Whether adults are detected
        """
        current_time = time.time()

        if has_children and not has_adults:
            # Only children detected
            if self.children_only_start_time is None:
                self.children_only_start_time = current_time
                print("⚠️  Children-only detection started")

            # Check if warning threshold is reached
            elapsed_time = current_time - self.children_only_start_time
            if elapsed_time >= self.warning_threshold:
                if not self.warning_active:
                    self.warning_active = True
                    print(f"🚨 WARNING: Unattended children detected for {elapsed_time:.1f} seconds!")



                # Send periodic notifications based on cooldown
                if (self.last_notification_time is None or
                        current_time - self.last_notification_time >= self.notification_cooldown):
                    self._send_azure_notification(elapsed_time)
                    self.last_notification_time = current_time

        else:
            # Reset if adults are present or no children detected
            if self.children_only_start_time is not None:
                # Send resolution notification if warning was active
                if self.warning_active and self.azure_sender:
                    elapsed_time = current_time - self.children_only_start_time
                    if has_adults and has_children:
                        resolution_message = f"RESOLVED: Adult supervision restored after {elapsed_time:.0f} seconds"
                        print("✅ Adult supervision restored")
                    elif not has_children:
                        resolution_message = f"RESOLVED: No children detected after {elapsed_time:.0f} seconds"
                        print("✅ No children detected")

                    self.azure_sender.send_notification(
                        resolution_message,
                        alert_type="situation_resolved",
                        elapsed_time=elapsed_time
                    )

            self.children_only_start_time = None
            self.warning_active = False
            self.last_escalation_level = -1


    def _send_azure_notification(self, elapsed_time):
        """
        Send notification to Azure

        Args:
            elapsed_time (float): Time elapsed since detection started
        """
        if self.azure_sender:
            message = f"ALERT: Unattended children detected for {elapsed_time:.0f} seconds in vehicle monitoring area"
            self.azure_sender.send_notification(message, elapsed_time=elapsed_time)
        else:
            print("⚠️ Azure notifications not configured")

    def get_warning_status(self):
        """
        Get current warning status

        Returns:
            tuple: (warning_active, elapsed_time)
        """
        if self.children_only_start_time is None:
            return False, 0

        elapsed_time = time.time() - self.children_only_start_time
        return self.warning_active, elapsed_time


def detect_faces(image, face_net, confidence_threshold=0.5):
    """Detect faces in image using DNN model"""
    # Get image dimensions
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and get detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # Lists to store face locations and confidences
    faces = []
    face_confidences = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence of the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_threshold:
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes are within the dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Calculate width and height of the face
            width = endX - startX
            height = endY - startY

            # Add the face and confidence to the lists
            faces.append((startX, startY, width, height))
            face_confidences.append(float(confidence))

    return faces, face_confidences


def classify_age(face_img, age_model, img_size=105):
    """Classify age of detected face"""
    # Convert to grayscale
    gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Resize to required dimensions
    resized_img = cv2.resize(gray_img, (img_size, img_size))

    # Normalize pixel values to [0, 1]
    normalized_img = resized_img / 255.0

    # Add batch and channel dimensions
    input_img = np.expand_dims(normalized_img, axis=0)  # Add batch dimension
    input_img = np.expand_dims(input_img, axis=-1)  # Add channel dimension

    # Get prediction
    prediction = age_model.predict(input_img, verbose=0)[0]

    # Get class with highest probability
    # Index 0: Child, Index 1: Adult
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # Return True for adult, False for child
    is_adult = (predicted_class == 1)

    return is_adult, float(confidence)


def draw_warning_overlay(frame, warning_active, elapsed_time):
    """Draw warning overlay on the frame"""
    if warning_active:
        h, w = frame.shape[:2]

        # Create semi-transparent red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Warning text
        warning_text = f"⚠️ UNATTENDED CHILDREN ALERT - {elapsed_time:.0f}s"
        cv2.putText(frame, warning_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Flashing effect
        if int(elapsed_time * 2) % 2:  # Flash every 0.5 seconds
            cv2.putText(frame, "🚨 EMERGENCY SERVICES NOTIFIED 🚨", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Azure notification status
        cv2.putText(frame, "📡 Azure Notifications Active", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def process_video(video_source, face_net, age_model, output_path=None, img_size=105, warning_time=60, azure_url=None):
    """Process video with face detection and age classification"""
    # Initialize unattended children tracker with Azure notifications
    tracker = UnattendedChildrenTracker(warning_threshold=warning_time, azure_url=azure_url)

    # Initialize video capture
    cap = cv2.VideoCapture(video_source)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Clone the frame for display
        display_frame = frame.copy()

        # Detect faces in the frame
        faces, confidences = detect_faces(frame, face_net)

        # Count children and adults
        child_count = 0
        adult_count = 0

        # Process each detected face
        for ((x, y, w, h), conf) in zip(faces, confidences):
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y + h, x:x + w]

            # Skip if face ROI is empty (can happen at image boundaries)
            if face_roi.size == 0:
                continue

            # Classify the face as child or adult
            is_adult, age_conf = classify_age(face_roi, age_model, img_size)

            # Count demographics
            if is_adult:
                adult_count += 1
                label = "Adult"
                color = (0, 255, 0)  # Green for adults
            else:
                child_count += 1
                label = "Child"
                color = (0, 0, 255)  # Red for children

            # Draw the bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

            # Display the classification and confidence
            text = f"{label}: {age_conf:.2f}"
            cv2.putText(display_frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add face detection confidence
            face_text = f"Face: {conf:.2f}"
            cv2.putText(display_frame, face_text, (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update unattended children tracker
        has_children = child_count > 0
        has_adults = adult_count > 0
        tracker.update_detection(has_children, has_adults)

        # Get warning status and draw overlay
        warning_active, elapsed_time = tracker.get_warning_status()
        if warning_active:
            draw_warning_overlay(display_frame, warning_active, elapsed_time)

        # Display counts and status
        status_text = f"Children: {child_count} | Adults: {adult_count}"
        cv2.putText(display_frame, status_text, (10, frame_height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if has_children and not has_adults and not warning_active:
            timer_text = f"Unattended timer: {elapsed_time:.1f}s"
            cv2.putText(display_frame, timer_text, (10, frame_height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Azure connection status
        azure_status = "Azure: Connected" if azure_url else "Azure: Disabled"
        cv2.putText(display_frame, azure_status, (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if azure_url else (0, 0, 255), 2)

        # Display frame count
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write the frame to the output video if specified
        if out:
            out.write(display_frame)

        # Display the frame
        cv2.imshow("Enhanced Child Safety Monitor", display_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


def process_image(image_path, face_net, age_model, output_path=None, img_size=105, azure_url=None):
    """Process single image with face detection and age classification"""
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Initialize Azure sender for image processing
    azure_sender = AzureNotificationSender(azure_url) if azure_url else None

    # Detect faces in the image
    faces, confidences = detect_faces(image, face_net)

    # Count children and adults
    child_count = 0
    adult_count = 0

    # Process each detected face
    for ((x, y, w, h), conf) in zip(faces, confidences):
        # Extract the face ROI (Region of Interest)
        face_roi = image[y:y + h, x:x + w]

        # Skip if face ROI is empty
        if face_roi.size == 0:
            continue

        # Classify the face as child or adult
        is_adult, age_conf = classify_age(face_roi, age_model, img_size)

        # Count and set display properties
        if is_adult:
            adult_count += 1
            label = "Adult"
            color = (0, 255, 0)  # Green for adults
        else:
            child_count += 1
            label = "Child"
            color = (0, 0, 255)  # Red for children

        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Display the classification and confidence
        text = f"{label}: {age_conf:.2f}"
        cv2.putText(image, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add face detection confidence
        face_text = f"Face: {conf:.2f}"
        cv2.putText(image, face_text, (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display summary
    h, w = image.shape[:2]
    summary_text = f"Children: {child_count} | Adults: {adult_count}"
    cv2.putText(image, summary_text, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Warning for image analysis and Azure notification
    if child_count > 0 and adult_count == 0:
        cv2.putText(image, "⚠️ UNATTENDED CHILDREN DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Send Azure notification for image analysis
        if azure_sender:
            message = f"IMAGE ANALYSIS: Unattended children detected in image {os.path.basename(image_path)}"
            azure_sender.send_notification(message, alert_type="image_analysis")

    # Save the output image if specified
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Output saved to {output_path}")

    # Display the image
    cv2.imshow("Enhanced Child Safety Monitor", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Enhanced Child Safety Monitor with Azure Notifications')
    parser.add_argument('--face-prototxt', required=True, help='Path to face detection prototxt')
    parser.add_argument('--face-model', required=True, help='Path to face detection model')
    parser.add_argument('--age-model', required=True, help='Path to age classification model (TensorFlow .h5 file)')
    parser.add_argument('--input', required=True,
                        help='Path to input image or video file, or camera index (0 for webcam)')
    parser.add_argument('--output', help='Path to output file (optional)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum probability to filter weak detections')
    parser.add_argument('--img-size', type=int, default=105, help='Image size for age classification model')
    parser.add_argument('--warning-time', type=int, default=60,
                        help='Time in seconds before unattended warning (default: 60)')
    parser.add_argument('--azure-url', type=str,
                        default="https://ai321.blob.core.windows.net/childnotofication?sp=racwdli&st=2025-07-03T17:21:56Z&se=2025-08-01T01:21:56Z&spr=https&sv=2024-11-04&sr=c&sig=kIZJ%2BfAk7tYC7tQdTItm3tTWBcAFFruD7t6iJ2g3Ojc%3D",
                        help='Azure Blob Storage URL for notifications')

    args = parser.parse_args()

    print(" Child Safety Monitoring System with Azure Integration")
    print(f" Warning threshold: {args.warning_time} seconds")
    print(f"Azure notifications: {'Enabled' if args.azure_url else 'Disabled'}")

    # Load the face detection model
    print("Loading face detection model...")
    face_net = cv2.dnn.readNet(args.face_model, args.face_prototxt)

    # Load the age classification model
    print("Loading age classification model...")
    age_model = tf.keras.models.load_model(args.age_model)

    # Check if input is an image or video
    if args.input.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Process image
        print(f"Processing image: {args.input}")
        process_image(args.input, face_net, age_model, args.output, args.img_size, args.azure_url)
    else:
        # Process video (file or webcam)
        try:
            video_source = int(args.input)
            print(f"Processing webcam feed from camera index {video_source}")
        except ValueError:
            video_source = args.input
            print(f"Processing video file: {video_source}")

        process_video(video_source, face_net, age_model, args.output, args.img_size,
                      args.warning_time, args.azure_url)

    print("Processing complete.")


if __name__ == "__main__":
    main()

# Usage examples:
#
# For video processing with Azure notifications:
# python enhanced_monitor.py --face-prototxt deploy.prototxt.txt --face-model res10_300x300_ssd_iter_140000.caffemodel --age-model best_model_fold_1.h5 --input VID-20250503-WA0084.mp4 --output monitored_output.mp4 --warning-time 60
#
# For webcam monitoring:
# python enhance