/**
 * FSR Sensors Header File
 * Include this in other files to access sensor functions
 */

#ifndef FSR_SENSORS_H
#define FSR_SENSORS_H

// Initialize sensors (call once in setup)
void initSensors();

// Read all sensors (updates values)
void readAllSensors();

// Get individual sensor values
int getFSR1();
int getFSR2();
int getFSR3();

// Get all sensors at once
void getAllSensors(int* sensor1, int* sensor2, int* sensor3);

// Print sensor values to Serial
void printSensors();

#endif
