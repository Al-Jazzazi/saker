/**
 * Simple ESP32 FSR Sensor Reader
 * Read three FSR sensors with easy-to-call functions
 */

// Pin Configuration - Update these based on your wiring
const int FSR1_PIN = 34;
const int FSR2_PIN = 35;
const int FSR3_PIN = 32;

// Settings
const int BAUD_RATE = 115200;
const int READ_DELAY = 100;  // milliseconds between readings

// Global variables to store sensor values
int fsr1_value = 0;
int fsr2_value = 0;
int fsr3_value = 0;

/**
 * Initialize sensors
 * Call this once in setup()
 */
void initSensors() {
  pinMode(FSR1_PIN, INPUT);
  pinMode(FSR2_PIN, INPUT);
  pinMode(FSR3_PIN, INPUT);
  analogReadResolution(12);  // 12-bit ADC (0-4095)
  Serial.begin(BAUD_RATE);
  Serial.println("FSR Sensors initialized");
}

/**
 * Read all three sensors
 * Call this to update all sensor values
 */
void readAllSensors() {
  fsr1_value = analogRead(FSR1_PIN);
  fsr2_value = analogRead(FSR2_PIN);
  fsr3_value = analogRead(FSR3_PIN);
}

/**
 * Get FSR1 value
 * Returns: int (0-4095)
 */
int getFSR1() {
  return fsr1_value;
}

/**
 * Get FSR2 value
 * Returns: int (0-4095)
 */
int getFSR2() {
  return fsr2_value;
}

/**
 * Get FSR3 value
 * Returns: int (0-4095)
 */
int getFSR3() {
  return fsr3_value;
}

/**
 * Get all sensor values at once
 * Parameters: pointers to store values
 */
void getAllSensors(int* sensor1, int* sensor2, int* sensor3) {
  *sensor1 = fsr1_value;
  *sensor2 = fsr2_value;
  *sensor3 = fsr3_value;
}

/**
 * Print all sensor values to Serial
 */
void printSensors() {
  Serial.print("FSR1: ");
  Serial.print(fsr1_value);
  Serial.print(" | FSR2: ");
  Serial.print(fsr2_value);
  Serial.print(" | FSR3: ");
  Serial.println(fsr3_value);
}

// ========== SETUP ==========
void setup() {
  initSensors();
  delay(1000);
}

// ========== MAIN LOOP ==========
void loop() {
  // Read all sensors
  readAllSensors();

  // Print values
  printSensors();

  // Optional: Access individual sensors
  // int sensor1 = getFSR1();
  // int sensor2 = getFSR2();
  // int sensor3 = getFSR3();

  delay(READ_DELAY);
}
