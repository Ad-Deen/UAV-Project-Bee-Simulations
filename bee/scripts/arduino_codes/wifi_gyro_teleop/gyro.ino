// Simulated gyro function — replace with real IMU code

// float getAccX() const { return a[0]; }
//     float getAccY() const { return a[1]; }
//     float getAccZ() const { return a[2]; }
//     float getGyroX() const { return g[0]; }
//     float getGyroY() const { return g[1]; }
//     float getGyroZ() const { return g[2]; }
void readGyro() {
  if (mpu.update()) {
        // static uint32_t prev_ms = millis();
        // if (millis() > prev_ms + 25) {
            
            gyroX = mpu.getRoll() + 179.0 + 1.28 ;
            if(gyroX > 60){
                gyroX = gyroX -360;
            }
            gyroX = filterX.update(gyroX);
            gyroY = filterY.update(mpu.getPitch());
            // gyroY = -mpu.getPitch() +0.42 ;
            // gyroZ = mpu.getYaw();
            // Serial.print("Roll, Pitch, Yaw: ");
            // Serial.print(gyroX);
            // Serial.print(", ");
            // Serial.print(gyroY);
            // Serial.print(", ");
            // Serial.println(gyroZ);
            // prev_ms = millis();
        
    }
}