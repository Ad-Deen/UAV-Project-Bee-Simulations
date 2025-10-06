void updatePID() {
  unsigned long now = millis();
  float dt = (now - lastUpdate) / 1000.0;
  if (dt < 0.001) return;  // Avoid divide-by-zero or noise spikes

  if (dt >= pidInterval / 1000.0) {
    lastUpdate = now;

    // === GYRO X PID ===
    float errorX = targetGyroX - gyroX;
    integralX += errorX * dt;
    integralX = constrain(integralX, -30, 30);  // Anti-windup
    float derivativeX = (errorX - previous_errorX) / dt;

    // if (abs(errorX) < 10) {
      correctionX = kpX * errorX + kiX * integralX + kdX * derivativeX;
    // } else {
    //   correctionX = kpX * errorX + kiX * integralX;
    // }
    correctionX = constrain(correctionX, -50, 50);
    previous_errorX = errorX;

    // === GYRO Y PID ===
      float errorY = targetGyroY - gyroY;
      integralY += errorY * dt;
      integralY = constrain(integralY, -20, 20);
      // Estimate angular velocity (derivative of pitch angle)
      float derivativeY = (errorY - previous_errorY) / dt;
      // Apply scaled derivative only when close to target
      float selective = constrain(10/(abs(errorY-5)*abs(errorY+5)*0.067+1)*5, 0 , 1);
      correctionY = kpY * errorY + kiY * integralY + kdY * derivativeY* selective;

      correctionY = constrain(correctionY, -90, 90);
      rev_correctionY = correctionY*2.0;
      previous_errorY = errorY;

    // Serial.print("errorY ");
    // Serial.print(errorY);
    // Serial.print(" - dervative ");
    // Serial.println(derivativeY );
    // Serial.print(" - brake ");
    // Serial.println(mpu.getGyroY());
    // Serial.print(" - CY ");
    // Serial.println(correctionY);
  }
}