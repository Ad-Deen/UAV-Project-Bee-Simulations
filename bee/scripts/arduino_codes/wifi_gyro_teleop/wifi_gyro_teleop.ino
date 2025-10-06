#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <ESP32Servo.h>
#include "MPU9250.h"

MPU9250 mpu;

//filter parameters
struct MovingAverageFilter {
  float buffer[10];
  int index;
  int count;

  MovingAverageFilter() : index(0), count(0) {
    for (int i = 0; i < 10; i++) buffer[i] = 0;
  }

  float update(float value) {
    buffer[index] = value;
    index = (index + 1) % 10;
    if (count < 10) count++;

    float sum = 0;
    for (int i = 0; i < count; i++) {
      sum += buffer[i];
    }
    return sum / count;
  }
};

MovingAverageFilter filterX;
MovingAverageFilter filterY;

static const int servoPin1 = 11; //rotor 1
static const int servoPin2 = 12; //rotor 4
static const int servoPin3 = 13; //rotor 2
static const int servoPin4 = 14; //rotor 3
static const int failsafe_timeout = 2000; //Rotor goes off after 2 seconds of no UDP comm
// PID tuning
// gyroX stable 1.6, 0.57 , 0.32
// gyroY stable 1.8, 0.7 , 0.7
float kpX = 0.0, kiX = 0.0, kdX = 0;
float kpY = 0.0, kiY = 0.0, kdY = 0;

// float delta = 0.1;
// State variables for gyroX
float integralX = 0;
float previous_errorX = 0;


// State variables for gyroY
float integralY = 0;
float previous_errorY = 0;

float correctionX = 0;
float correctionY = 0;
float rev_correctionX = 0;
float rev_correctionY = 0;

int targetGyroX= 0 ;
int targetGyroY = 0 ;

unsigned long lastUpdate = 0;
unsigned long pidInterval = 5;  // 200 Hz
uint32_t lastPwmWrite = 0 ;
unsigned long lastUdpReceiveTime = 0;  // Track last time UDP packet was received

// Servo objects
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;

uint16_t r1_off=0 ,r2_off=0 ,r3_off=0 ,r4_off=0 ;

uint16_t r = 0;
uint16_t kp_transfer = 0;
uint16_t kd_transfer = 0;
uint16_t ki_transfer = 0;

float gyroX = 0.0, gyroY = 0.0, gyroZ = 0.0;
// Replace with your Wi-Fi credentials
// const char* ssid = "Hive";
// const char* password = "12345678";
const char* ssid = "Explore";
const char* password = "Explore.us";

WiFiUDP udp;
const unsigned int localPort = 4210;

uint8_t rotorCmd[8] = {0};      // Received motor speeds

IPAddress pcIP;
uint16_t pcPort;


void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  delay(1000);

  Serial.println("Calibrating ESC ...");

  // Attach the servos and set the PWM range (1000 - 2000 microseconds)
  servo1.attach(servoPin1, 1000, 2000);
  servo2.attach(servoPin2, 1000, 2000);
  servo3.attach(servoPin3, 1000, 2000);
  servo4.attach(servoPin4, 1000, 2000);
  delay(2000);

  // Set throttle to maximum (full throttle) for calibration
  servo1.writeMicroseconds(2000);
  servo2.writeMicroseconds(2000);
  servo3.writeMicroseconds(2000);
  servo4.writeMicroseconds(2000);
  Serial.println("Full throttle for calibration...");
  delay(2000);  // Wait for ESC to register maximum throttle

  // Set throttle to minimum (no throttle) to complete calibration
  servo1.writeMicroseconds(1000);
  servo2.writeMicroseconds(1000);
  servo3.writeMicroseconds(1000);
  servo4.writeMicroseconds(1000);
  Serial.println("Throttle set to minimum...");
  delay(2000);

  Serial.println("ESC Calibrated!");

  Wire.begin();
  delay(2000);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected. IP address: " + WiFi.localIP().toString());

  udp.begin(localPort);
  Serial.println("Listening on UDP port " + String(localPort));

  

  if (!mpu.setup(0x68)) {  // change to your own address
      while (1) {
          digitalWrite(LED_BUILTIN, HIGH);
          Serial.println("MPU connection failed. Please check your connection with `connection_check` example.");
          delay(1000);
          digitalWrite(LED_BUILTIN, LOW);
          delay(1000);
      }
  }
  Serial.println("MPU connection success!!");
  digitalWrite(LED_BUILTIN, LOW);
}

void loop() {
  
  // lastPwmWrite = millis();  
  // Read gyro
    readGyro();
    // stabilization();
    updatePID();
    // Serial.print("CorrectX = ");
    // Serial.print(correctionX);
    // Serial.print("  CorrectY = ");
    // Serial.println(correctionY);
    if(correctionY >0){
    // lastPwmWrite = millis();  
      // Set motors here using rotorCmd[0]..[3] (actual PWM logic omitted)
      r1_off = constrain(r + (int)correctionX - (int)rev_correctionY, 0 , 300);
      r2_off = constrain(r - (int)correctionX - (int)rev_correctionY, 0 , 300);
      r3_off = constrain(r - (int)correctionX + (int)correctionY, 0 , 300);
      r4_off = constrain(r + (int)correctionX + (int)correctionY, 0 , 300);
    }else{
      r1_off = constrain(r + (int)correctionX - (int)correctionY, 0 , 300);
      r2_off = constrain(r - (int)correctionX - (int)correctionY, 0 , 300);
      r3_off = constrain(r - (int)correctionX + (int)rev_correctionY, 0 , 300);
      r4_off = constrain(r + (int)correctionX + (int)rev_correctionY, 0 , 300);
    }
  // Failsafe: shut down if no signal for 2 seconds
if (millis() - lastUdpReceiveTime > failsafe_timeout) {
  // Serial.println(millis() - lastUdpReceiveTime);
  r1_off = 0;
  r2_off = 0;
  r3_off = 0;
  r4_off = 0;


  // Serial.println("Failsafe triggered — no UDP for 2 sec!");
  delay(20);  // Just to reduce UART spamming
  // return;     // Exit loop early to skip motor write
}

  if (millis() - lastPwmWrite > 20) {
    // Serial.println(millis() - lastPwmWrite);
    lastPwmWrite = millis();  
    servo1.writeMicroseconds(1000 + r1_off);
    servo2.writeMicroseconds(1000 + r2_off);
    servo3.writeMicroseconds(1000 + r3_off);
    servo4.writeMicroseconds(1000 + r4_off);
  }

  int packetSize = udp.parsePacket();
  if (packetSize == 8) {  // 4 rotor values × 2 bytes (uint16_t)
  lastUdpReceiveTime = millis();  // Update last valid communication time

    // Serial.println("*******************Wifi gained***************");
    udp.read(rotorCmd, 8);

    // Decode into global rotor variables
    memcpy(&r, rotorCmd, 2);
    memcpy(&kp_transfer, rotorCmd + 2, 2);
    memcpy(&kd_transfer, rotorCmd + 4, 2);
    memcpy(&ki_transfer, rotorCmd + 6, 2);

    pcIP = udp.remoteIP();
    pcPort = udp.remotePort();

    //adjust PID parameters
    kpY = (float)kp_transfer/100 ;
    kdY = (float)kd_transfer/100;
    kiY = (float)ki_transfer/100;
    // Prepare response
    uint8_t response[20];
    memcpy(response,     &r1_off, 2);
    memcpy(response + 2, &r2_off, 2);
    memcpy(response + 4, &r3_off, 2);
    memcpy(response + 6, &r4_off, 2);
    memcpy(response + 8,  &gyroX, 4);
    memcpy(response + 12, &gyroY, 4);
    memcpy(response + 16, &gyroZ, 4);
    // Send response
    udp.beginPacket(pcIP, pcPort);
    udp.write(response, 20); // 4 bytes + 4 + 6 = 16 total
    udp.endPacket();
  }
  
    // for X
  // Serial.print("kpX kdX kiX : ");
  // Serial.print(kpX);
  // Serial.print(" ");
  // Serial.print(kdX);
  // Serial.print(" ");
  // Serial.print(kiX);
  // Serial.print(" ");
  // Serial.println((int)correctionX);
  // for Y
  // Serial.print("kpY kdY kiY : ");
  // Serial.print(kpY);
  // Serial.print(" ");
  // Serial.print(kdY);
  // Serial.print(" ");
  // Serial.print(kiY);
  // Serial.print(" ");
  // Serial.println((int)correctionY);
  
  // Serial.println("-----------------------Wifi missed ----------------");
  // delayMicroseconds(10);
  // Serial.println(millis() - lastPwmWrite);
}
