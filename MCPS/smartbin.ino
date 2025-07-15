#include "HX711.h"
#include <Servo.h>
#include <SoftwareSerial.h>
#include <SPI.h>
#include <MFRC522.h>

/*
 * Arduino's pins
 * 
 * D2 RX ESP8266
 * D3 TX ESP8266
 * D4 CLK HX711
 * D5 DOUT HX711
 * D6 SERVOMOTOR PULSE
 * D7 RST RFID
 * D8 SDA RFID
 * D9 ECHO ULTRASOUND SENSOR
 * D10 TRIG ULTRASOUND SENSOR
 * D11 MOSI RFID
 * D12 MISO RFID
 * D13 SCK RFID
 * D14 BLUE RGBLED
 * D15 GREEN RGBLED
 * D16 RED RGBLED
 * 
 * ESP8266 CH-EN 3.3V
 * 
 */

#define BIN_ID 1
#define BIN_HEIGHT 50 //IN MM
#define CAPACITY_THRESHOLD 20 //While dealing with small distances the ultrasound sensor loses in precision thus the sensor will treat anything above the treshold as a completely full bin (70% -> 100%)

#define TX 2
#define RX 3
#define SCALE_CLK 4
#define SCALE_DT 5 
#define SERVO_PULSE 6 
#define RST_PIN 7
#define SS_PIN 8
#define ULTRA_ECHO 9
#define ULTRA_TRIG 10
#define RGB_BLUE 14 
#define RGB_GREEN 15 //(SCAMBIA ROSSO CON VERDE)
#define RGB_RED 16

MFRC522 mfrc522(SS_PIN, RST_PIN);  //Istance of the RFID library
SoftwareSerial esp8266(RX,TX);  //Simulate the serial monitor utilized while interacting with the ESP
Servo binCover;

HX711 scale(SCALE_DT, SCALE_CLK); 
float calibration_factor = 835.30; //NEEDS CALIBRATION ONCE PHYSICALLY ATTACHED TO THE BIN
float units;

boolean binFull;
String UID = "";

unsigned char check_connection = 0; 
unsigned char times_check = 0; 
boolean error;
String AP = "HomeLife"; 
String PASS = "GiuliaMarco";
//String AP = "Vodafone-33986335"; 
//String PASS = "#s1433zmTpq77ax"; 
//String HOST = "192.168.1.139";
String HOST = "192.168.1.139";
String PORT = "8080";
int countTrueCommand;
int countTimeCommand; 
boolean found = false; 

String admins = "_99_F0_FD_D5"; 
String users = "_99_F0_FD_D5,_07_B1_89_44,_C9_B9_9C_96,_49_4B_F9_D5";

void connectWifi(){
  esp8266.println("AT+RST");
  delay(2000);
  Serial.println("Connecting to wifi");
  while (check_connection == 0)
  {
    Serial.print(".");
    esp8266.println("AT+CWJAP=\"" + AP + "\",\"" + PASS + "\"");
    esp8266.setTimeout(5000);
    if(esp8266.find("WIFI CONNECTED\r\n")==1)
    {
      Serial.println("WIFI CONNECTED");
      break;
    }
    times_check++;
    if(times_check > 3)
    {
      times_check = 0;
      Serial.println("Trying to reconnect...");
    }
  }
}

void sendData(String UID, float weight) { 
  Serial.println();
  Serial.println("User " + UID + " put into the bin " + weight + " grams of garbage \nCapacity at: " + measureCapacity());
  Serial.println();
  String getData = "GET  /test?user_id='" + UID + "'&bin_id=" + BIN_ID + "&weight=" + String(int(weight))+"&capacity="+String(measureCapacity());
  sendCommand("AT+CIPMUX=1",5,"OK");
  sendCommand("AT+CIPSTART=0,\"TCP\",\""+ HOST +"\","+ PORT,15,"OK");
  sendCommand("AT+CIPSEND=0," +String(getData.length()),4,">");
  esp8266.println(getData);delay(1500);countTrueCommand++;
  sendCommand("AT+CIPCLOSE=0",5,"OK"); 
}

void sendCommand(String command, int maxTime, char readReplay[]) { //Sends requests to the ESP8266 //restore comments for debugging
  Serial.print(countTrueCommand);
  Serial.print(". at command => ");
  Serial.print(command);
  Serial.print(" ");
  while(countTimeCommand < (maxTime*1))
  {
    esp8266.println(command);//at+cipsend
    if(esp8266.find(readReplay))//ok
    {
      found = true;
      break;
    }
    countTimeCommand++;
  }
  if(found == true)
  {
    Serial.println("OYI");
    countTrueCommand++;
    countTimeCommand = 0;
  }
  if(found == false)
  {
    Serial.println("Fail");
    countTrueCommand = 0;
    countTimeCommand = 0;
  }
  found = false;
 }

long measureCm(){
  // The sensor is triggered by a HIGH pulse of 10 or more microseconds.
  // Give a short LOW pulse beforehand to ensure a clean HIGH pulse:
  digitalWrite(ULTRA_TRIG, LOW);
  delayMicroseconds(5);
  digitalWrite(ULTRA_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRA_TRIG, LOW);
  // Read the signal from the sensor: a HIGH pulse whose
  // duration is the time (in microseconds) from the sending
  // of the ping to the reception of its echo off of an object.
  pinMode(ULTRA_ECHO, INPUT);
  long duration = pulseIn(ULTRA_ECHO, HIGH);
  // Convert the time into a distance
  long cm = (duration/2) / 2.91;     // Divide by 29.1 or multiply by 0.0343
  return cm;
}

long measureCapacity(){
  long cm = measureCm();
  float binCapacity = (BIN_HEIGHT - cm);
  binCapacity = binCapacity/BIN_HEIGHT;
  binCapacity = binCapacity*100;
  if (binCapacity < 0)
    return 0;
  if (binCapacity >= CAPACITY_THRESHOLD)
    return 100;
  return binCapacity;
}

float measureWeight(){
  float currentWeight = scale.get_units(20);  // get average of 20 scale readings
  if (currentWeight < 0) {
    currentWeight = 0.00;
  }
  scale.tare();
  return currentWeight;
}

void rgbColor(unsigned char red, unsigned char green, unsigned char blue){
  analogWrite(RGB_RED, red);
  analogWrite(RGB_GREEN, green);
  analogWrite(RGB_BLUE, blue);
}

void openBin(Servo binCover){
  binCover.write(0); //open the cap of the bin
}

void closeBin(Servo binCover){
  binCover.write(80); //open the cap of the bin
}

String rfid_scan(){
  // Look for new cards
  if ( ! mfrc522.PICC_IsNewCardPresent())
  {
    return "";
  }
  // Select one of the cards
  if ( ! mfrc522.PICC_ReadCardSerial())
  {
    return "";
  }
  //Show UID on serial monitor
  Serial.print("\nUID tag : ");
  String UID = "";
  
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    Serial.print(mfrc522.uid.uidByte[i] < 0x10 ? "_0" : "_");
    UID.concat(mfrc522.uid.uidByte[i] < 0x10 ? "_0" : "_");
    Serial.print(mfrc522.uid.uidByte[i], HEX);
    UID.concat(String(mfrc522.uid.uidByte[i], HEX));
  }
  return UID;
 }

 void binInteraction(String UID){
  Serial.println("\nAcces authorized\n");
  rgbColor(0, 255, 0); //green
  openBin(binCover);
  Serial.println("\nYou have 10 seconds to insert your wastes\n");
  Serial.println();
  delay(10000); //during this time the user should insert his garbage
  closeBin(binCover);
  float weight = measureWeight();
  sendData(UID, weight);
}



void setup() {
  // put your setup code here, to run once:
  
  //Serial Port begin
  Serial.begin(9600);
  esp8266.begin(115200);
  SPI.begin();
  mfrc522.PCD_Init(); /* It initialize the RFID reader */  

  connectWifi();
  
  pinMode(ULTRA_TRIG, OUTPUT);
  pinMode(ULTRA_ECHO, INPUT);
  
  pinMode(RGB_BLUE, OUTPUT);
  pinMode(RGB_GREEN, OUTPUT);
  pinMode(RGB_RED, OUTPUT);

  binCover.attach(SERVO_PULSE);

  scale.set_scale(calibration_factor);
  scale.tare();  //Reset the scale to 0  
  binFull = (measureCapacity() == 100);

  Serial.print("\n\nThis is SmartBin\n");
  delay(100);
  Serial.println("Capacity is at " + String(measureCapacity()));
  delay(100);
}

void loop() {
  // put your main code here, to run repeatedly:

  if(binFull)
  {
    Serial.println("\nThe bin is full, waiting for an admin: ");
    Serial.println();
    rgbColor(255, 215, 0); //yellow
    String UID = rfid_scan();
    while (UID == "") {UID = rfid_scan();}
    UID.trim();
    UID.toUpperCase();
    if(admins.indexOf(UID) == -1){
      Serial.println("\nThe username is incorrect\n");
      rgbColor(255, 0, 0); //red
      delay(3000);
      return;
      }
      binInteraction(UID);  
  }else{
    Serial.println("\nApproximate you card: ");
    Serial.println();
    rgbColor(0, 0, 255); //blue
    
    String UID = rfid_scan();
    while (UID == "") {UID = rfid_scan();}
    UID.trim();
    UID.toUpperCase();
    if( users.indexOf(UID) == -1){
      Serial.println("\nThe username is incorrect\n");
      rgbColor(255, 0, 0); //red
      delay(3000);
      return;
      }
      binInteraction(UID);
  }
  binFull = (measureCapacity() == 100);
  UID="";
}
