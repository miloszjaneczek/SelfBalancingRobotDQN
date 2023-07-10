#include <Wire.h>
#include "Motor.h"
#define AIN1 13
#define AIN2 12
#define BIN1 8
#define BIN2 7
#define PWMA 6
#define PWMB 3
////////////////VARIABLE DEFINATION///////////////
int mspeed = 10;
Motor *motorA, *motorB;
int16_t Acc_rawX, Acc_rawY, Acc_rawZ,Gyr_rawX, Gyr_rawY, Gyr_rawZ;
float Acceleration_angle[2];
float Acceleration_angleb[2]; //BIAS FOR CORRECTION
float Gyro_angle[2];
float Gyro_angleb[2]; //BIAS FOR CORRECTION
float Total_angle[2];
float elapsedTime, time, timePrev;
int speedDir = 1;
int countPID = 0;
float rad_to_deg = 180/3.141592654;
float PID, error, previous_error;
float pid_p=0;
float pid_i=0;
float pid_d=0;
String str, leftMotor, rightMotor;
////////////////////////PID CONSTANST/////////////////////
float pidToggle = 10; //WHEN TO ACTIVATE PID
float kp=19;//25;//;
float ki=0;//5;
float kd=0.15;//0.6;//
float desired_angle = 0;//////////////TARGET ANGLE/////////////
void setup() 
{
  motorA = new Motor(AIN1, AIN2, PWMA);
  motorB = new Motor(BIN2, BIN1, PWMB);
  Wire.begin(); /////////////TO BEGIN I2C COMMUNICATIONS///////////////
  Wire.beginTransmission(0x68);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  Wire.beginTransmission(0x68);
    Wire.write(0x3B); 
    Wire.endTransmission(false);
    Wire.requestFrom(0x68,6,true);
    ////////////////////PULLING RAW ACCELEROMETER DATA FROM IMU///////////////// 
    Acc_rawX=Wire.read()<<8|Wire.read(); 
    Acc_rawY=Wire.read()<<8|Wire.read();
    Acc_rawZ=Wire.read()<<8|Wire.read(); 
    /////////////////////CONVERTING RAW DATA TO ANGLES/////////////////////
    Acceleration_angleb[0] = atan(((Acc_rawY/16384.0))/sqrt(pow(((Acc_rawX/16384.0)),2) + pow(((Acc_rawZ/16384.0)),2)))*rad_to_deg;
    Acceleration_angleb[1] = atan(-1*((Acc_rawX/16384.0))/sqrt(pow(((Acc_rawY/16384.0)),2) + pow(((Acc_rawZ/16384.0)),2)))*rad_to_deg;

    Gyro_angleb[0] = Gyr_rawX/131.0; 
    Gyro_angleb[1] = Gyr_rawY/131.0;
  Serial.begin(115200);
  time = micros()/1000; ///////////////STARTS COUNTING TIME IN MILLISECONDS/////////////
}
void loop() 
{
    timePrev = time;  
    time = micros()/1000;  
    elapsedTime = (time - timePrev) / 1000; 
    Wire.beginTransmission(0x68);
    Wire.write(0x3B); 
    Wire.endTransmission(false);
    Wire.requestFrom(0x68,6,true);
    ////////////////////PULLING RAW ACCELEROMETER DATA FROM IMU///////////////// 
    Acc_rawX=Wire.read()<<8|Wire.read(); 
    Acc_rawY=Wire.read()<<8|Wire.read();
    Acc_rawZ=Wire.read()<<8|Wire.read(); 
    /////////////////////CONVERTING RAW DATA TO ANGLES/////////////////////
    Acceleration_angle[0] = atan(((Acc_rawY/16384.0))/sqrt(pow(((Acc_rawX/16384.0)),2) + pow(((Acc_rawZ/16384.0)),2)))*rad_to_deg;
    Acceleration_angle[1] = atan(-1*((Acc_rawX/16384.0))/sqrt(pow(((Acc_rawY/16384.0)),2) + pow(((Acc_rawZ/16384.0)),2)))*rad_to_deg;
    Wire.beginTransmission(0x68);
    Wire.write(0x43);
    Wire.endTransmission(false);
    Wire.requestFrom(0x68,4,true); 
    //////////////////PULLING RAW GYRO DATA FROM IMU/////////////////////////
    Gyr_rawX=Wire.read()<<8|Wire.read(); 
    Gyr_rawY=Wire.read()<<8|Wire.read(); 
    ////////////////////CONVERTING RAW DATA TO ANGLES///////////////////////
    Gyro_angle[0] = Gyr_rawX/131.0; 
    Gyro_angle[1] = Gyr_rawY/131.0;
    //////////////////////////////COMBINING BOTH ANGLES USING COMPLIMENTARY FILTER////////////////////////
    Acceleration_angle[0] = Acceleration_angle[0] - Acceleration_angleb[0];
    Acceleration_angle[1] = Acceleration_angle[1] - Acceleration_angleb[1];
    Gyro_angle[0] = Gyro_angle[0] - Gyro_angleb[0];
    Gyro_angle[1] = Gyro_angle[1] - Gyro_angleb[1];
    Total_angle[0] = 0.99 *(Total_angle[0] + Gyro_angle[0]*elapsedTime) + 0.01*Acceleration_angle[0];
    Total_angle[1] = 0.98 *(Total_angle[1] + Gyro_angle[1]*elapsedTime) + 0.02*Acceleration_angle[1];
    ////TOTAL_ANGLE[0] IS THE PITCH ANGLE WHICH WE NEED////////////
    error = Total_angle[0] - desired_angle; /////////////////ERROR CALCULATION////////////////////
    ///////////////////////PROPORTIONAL ERROR//////////////
    pid_p = kp*error;
    ///////////////////////INTERGRAL ERROR/////////////////
    pid_i = pid_i+(ki*error);  
    ///////////////////////DIFFERENTIAL ERROR//////////////
    pid_d = kd*((error - previous_error)/elapsedTime);
    ///////////////////////TOTAL PID VALUE/////////////////
    PID = pid_p + pid_d;
    ///////////////////////UPDATING THE ERROR VALUE////////
    previous_error = error;
    ///////////////////////CONTROL THROUGH SERIAL//////////
    if(abs(error) < pidToggle && countPID >= 1000){
      Serial.setTimeout(10000);
      if(Total_angle[0]>=0)
      {
        mspeed = abs(PID);
        if (mspeed > 255) mspeed = 255;
        
      }
      if(Total_angle[0]<0)
      {
        mspeed = -abs(PID);
        if(mspeed < -255) mspeed = -255;
      
      }
      if(Total_angle[0]>45)
        halt();
      if(Total_angle[0]<-45)
        halt();
      Serial.println(String(Total_angle[0]) + " " + String(mspeed) + " " + String(Gyro_angle[0]));
      str = Serial.readStringUntil('\n');
      mspeed = str.toInt();
      anti();
    }
    //////////CONTROL BY PID WHEN NOT TIMED OUT//////
    if(countPID < 1000){
      if(Total_angle[0]>=0)
      {
        if(abs(PID)>255)
        speedDir = 1;
          
        mspeed = abs(PID);
        countPID++;
        anti();
      }
      if(Total_angle[0]<0)
      {
        if(abs(PID)>255) speedDir = -1;
        mspeed = abs(PID);
        countPID++;
        clockw();
      }
      if(Total_angle[0]>45)
        halt();
      if(Total_angle[0]<-45)
        halt();
    
      Serial.println("PID");
    }
    /////////////RESET TIMEOUT/////////////
    if(abs(error) > pidToggle) countPID = 0;
    
}
//////////////MOVEMENT FUNCTION///////////////////
void clockw()
{
  motorA->set(mspeed);
  motorB->set(-mspeed);
}
void anti()
{

  motorA->set(-mspeed);
  motorB->set(mspeed);
}
void halt()
{
  
  motorA->set(0);
  motorB->set(0);
  
}

       