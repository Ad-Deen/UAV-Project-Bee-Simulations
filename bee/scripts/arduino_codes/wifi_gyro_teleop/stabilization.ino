void stabilization(){
  //Drone architecture
//              ^ gyroY (-30)
//              |
//        R4    |   R3
//         \    |    /
//          \   |   /
//           \  |  /
// gyroX(+30) \ | /     gyroX(-30)
//             \|/
//             /|\
//            / | \
//           /  |  \ 
//          /   |   \
//         /    |    \
//        R1    |    R2
//              |
//              v
//           gyroY (+30)
  updatePID();
  // Serial.print("CorrectX = ");
  // Serial.print(correctionX);
  // Serial.print("  CorrectY = ");
  // Serial.println(correctionY);
  // r3_off = (int)correctionX + (int)correctionY;
  // r2_off = (int)correctionX - (int)correctionY;
  // r1_off = - (int)correctionX - (int)correctionY;
  // r4_off = - (int)correctionX + (int)correctionY;
}