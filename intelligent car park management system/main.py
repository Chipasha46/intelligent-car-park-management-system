import time
import re
import requests
import numpy as np
import cv2
import pytesseract
from tensorflow.lite.python.interpreter import Interpreter
import RPi.GPIO as GPIO

# Define GPIO pins for the ultrasonic sensor
TRIG_PIN = 23
ECHO_PIN = 24

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# Initialize TensorFlow Lite interpreter for detection
modelpath_detection = '/home/pi/Desktop/Licence-Plate-Detection-using-TensorFlow-Lite/detect.tflite'
lblpath = 'labelmap.txt'
min_conf = 0.6  # Adjust as needed
min_char_area = 1500  # Adjust as needed

interpreter_detection = Interpreter(model_path=modelpath_detection)
interpreter_detection.allocate_tensors()
input_details_detection = interpreter_detection.get_input_details()
output_details_detection = interpreter_detection.get_output_details()
height_detection = input_details_detection[0]['shape'][1]
width_detection = input_details_detection[0]['shape'][2]

float_input_detection = (input_details_detection[0]['dtype'] == np.float32)

input_mean_detection = 127.5
input_std_detection = 127.5

with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)

# Server endpoint URL
server_endpoint = 'https://intelligentcarpark2023.00webhostapp.com'

# Add a delay for the ultrasonic sensor to settle
time.sleep(2)

prev_sensing_state = None  # Initialize previous sensing state
occupancy_updated = False  # Flag to track if occupancy state has been updated

while True:
    try:
        # Check if a car is detected by the ultrasonic sensor
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        pulse_start = time.time()
        pulse_end = time.time()

        while GPIO.input(ECHO_PIN) == 0:
            pulse_start = time.time()

        while GPIO.input(ECHO_PIN) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        distance = round(distance, 2)

        print(f"Distance: {distance} cm")

        if 50 < distance < 150:  # Adjust the range based on your environment
            current_sensing_state = 'Occupied'
        else:
            current_sensing_state = 'Vacant'

        # Define 'frame' outside the block to make it accessible later
        ret, frame = cap.read()

        if current_sensing_state != prev_sensing_state:
            if not occupancy_updated:  # Update only if occupancy state is not updated yet
                prev_sensing_state = current_sensing_state
                occupancy_updated = True  # Set the flag to True

                if current_sensing_state == 'Occupied':
                    print("Occupied - Sending update to the server...")

                    # Car detected, capture image and continue with license plate recognition
                    ret, frame = cap.read()

                    # Resize and process the image for license plate detection
                    imH, imW, _ = frame.shape
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_resized = cv2.resize(image_rgb, (width_detection, height_detection))
                    input_data = np.expand_dims(image_resized, axis=0)

                    if float_input_detection:
                        input_data = (np.float32(input_data) - input_mean_detection) / input_std_detection

                    interpreter_detection.set_tensor(input_details_detection[0]['index'], input_data)
                    
                    try:
                        interpreter_detection.invoke()
                    except Exception as e:
                        print(f"Error during TensorFlow Lite inference: {e}")
                        continue

                    # Get detection results
                    boxes = interpreter_detection.get_tensor(output_details_detection[1]['index'])[0]
                    classes = interpreter_detection.get_tensor(output_details_detection[3]['index'])[0]
                    scores = interpreter_detection.get_tensor(output_details_detection[0]['index'])[0]

                    vacant_detected = True  # Assume vacant initially

                    for i in range(len(scores)):
                        if min_conf < scores[i] <= 1.0:
                            ymin = int(max(1, (boxes[i][0] * imH)))
                            xmin = int(max(1, (boxes[i][1] * imW)))
                            ymax = int(min(imH, (boxes[i][2] * imH)))
                            xmax = int(min(imW, (boxes[i][3] * imW)))

                            char_area = (xmax - xmin) * (ymax - ymin)

                            if char_area > min_char_area:
                                # Process license plate
                                license_plate_crop = frame[ymin:ymax, xmin:xmax, :]
                                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                                license_plate_crop_blur = cv2.GaussianBlur(license_plate_crop_gray, (5, 5), 0)
                                _, license_plate_crop_thresh = cv2.threshold(
                                    license_plate_crop_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                                )
                                kernel = np.ones((3, 3), np.uint8)
                                license_plate_crop_thresh = cv2.morphologyEx(license_plate_crop_thresh, cv2.MORPH_CLOSE,
                                                                             kernel)
                                contours, _ = cv2.findContours(license_plate_crop_thresh, cv2.RETR_EXTERNAL,
                                                               cv2.CHAIN_APPROX_SIMPLE)
                                min_contour_area = 500
                                filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                                contour_img = np.zeros_like(license_plate_crop_thresh)
                                cv2.drawContours(contour_img, filtered_contours, -1, 255, thickness=cv2.FILLED)
                                license_plate_crop_thresh = cv2.bitwise_and(license_plate_crop_thresh, contour_img)

                                license_plate_text = pytesseract.image_to_string(
                                    license_plate_crop_thresh, config='--psm 11 --oem 3 -l eng'
                                )
                                license_plate_text = re.sub(r'[^a-zA-Z0-9]', '', license_plate_text)

                                if license_plate_text.strip() and len(license_plate_text) >= 3:
                                    print(f"Detected License Plate - {license_plate_text.strip()}")
                                    vacant_detected = False  # Occupied, not vacant

                                    # Send license plate information to the server
                                    payload = {'license_plate': license_plate_text, 'occupancy_status': 'Occupied'}
                                    response = requests.post(server_endpoint, data=payload)
                                    response.raise_for_status()  # Check for HTTP errors

                                    if response.status_code == 200:
                                        print("Data sent successfully!")
                                        occupancy_updated = False  # Reset the flag when data is sent
                                    else:
                                        print(f"Failed to send. Status code: {response.status_code}")

                    if vacant_detected:
                        print("No valid license plate detected - Assuming Vacant")

                    print("Capturing image...")
                    cv2.imwrite("captured_image.jpg", frame)
                    print("Image captured!")

                else:
                    print("Vacant - Sending update to the server...")
                    # Update server that the space is vacant
                    response = requests.post(server_endpoint, data={'occupancy_status': 'Vacant'})
                    response.raise_for_status()  # Check for HTTP errors

                    if response.status_code == 200:
                        print("Data sent successfully!")
                        occupancy_updated = False  # Reset the flag when data is sent
                    else:
                        print(f"Failed to send. Status code: {response.status_code}")

        else:
            occupancy_updated = False  # Reset the flag when occupancy state is the same

        print(f"Occupancy State: {current_sensing_state}")
        cv2.imshow('output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        print("Script terminated by user.")
        break

    except requests.exceptions.RequestException as e:
        print(f"Error during server communication: {e}")
        # Add a delay before retrying
        time.sleep(5)  # Adjust the delay time as needed

    except Exception as e:
        print(f"Unexpected error: {e}")

# Clean up GPIO
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()
