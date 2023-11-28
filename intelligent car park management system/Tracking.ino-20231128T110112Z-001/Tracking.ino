#include <TinyGPS++.h>
#include <HardwareSerial.h>
#include <WiFi.h>
#include <HTTPClient.h>

// Define the serial port for communication with the GPS module
HardwareSerial SerialGPS(2);

// Create a TinyGPS++ object
TinyGPSPlus gps;

// Define the interval for updating coordinates (5 seconds)
const unsigned long updateInterval = 5000;

unsigned long lastUpdateTime = 0;
double prevLatitude = 0.0;
double prevLongitude = 0.0;

const char *ssid = "Airtel-F66D";
const char *password = "01200049";

// Server endpoint
const char *serverEndpoint = "intelligentcarpark2023.000webhostapp.com";

// Vehicle ID to track
int vehicleID = 6; // Change this to the ID of your specific vehicle

void setup() {
  // Start the serial communication with the GPS module
  Serial.begin(9600);
  SerialGPS.begin(9600, SERIAL_8N1, 16, 17);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
}

void loop() {
  // Read data from the GPS module
  while (SerialGPS.available() > 0) {
    char c = SerialGPS.read();

    if (gps.encode(c)) {
      if (gps.location.isValid()) {
        double currentLatitude = gps.location.lat();
        double currentLongitude = gps.location.lng();

        // Check if the coordinates have changed
        if (currentLatitude != prevLatitude || currentLongitude != prevLongitude) {
          Serial.print("Latitude: ");
          Serial.print(currentLatitude, 6); // Latitude
          Serial.print(", Longitude: ");
          Serial.println(currentLongitude, 6); // Longitude

          // Update the previous coordinates
          prevLatitude = currentLatitude;
          prevLongitude = currentLongitude;

          if (millis() - lastUpdateTime >= updateInterval) {
            // Update the last update time
            lastUpdateTime = millis();

            // Send coordinates to the server only if the vehicle ID matches
            if (isTrackingVehicle()) {
              sendCoordinatesToServer(prevLatitude, prevLongitude, vehicleID);
            }
          }
        }
      }
    }
  }
}

void sendCoordinatesToServer(double latitude, double longitude, int vehicleID) {
  // Create an HTTP client object
  HTTPClient http;

  // Construct the URL for the Laravel endpoint
  String url = "https://" + String(serverEndpoint) + "/api/update-location";

  // Set up the POST request body with latitude, longitude, and vehicle ID
  String postBody = "latitude=" + String(latitude, 6) +
                    "&longitude=" + String(longitude, 6) +
                    "&car_id=" + String(vehicleID);

  // Begin the HTTP POST request
  http.begin(url);

  // Set the content type to application/x-www-form-urlencoded
  http.addHeader("Content-Type", "application/x-www-form-urlencoded");

  // Make the POST request and check for errors
  int httpCode = http.POST(postBody);
  Serial.print("HTTP Response Code: ");
  Serial.println(httpCode);

  if (httpCode > 0) {
    // If successful, print acknowledgment
    String response = http.getString();
    Serial.println("Server Response: " + response);
    Serial.println("Data sent successfully");
  } else {
    // If the request failed, print the error
    Serial.println("Failed to send data");
  }

  // End the request
  http.end();
}

bool isTrackingVehicle() {
  // Add logic here to determine if the current vehicle ID matches the one to be tracked
  // For now, always return true to send coordinates for demonstration purposes
  return true;
}
