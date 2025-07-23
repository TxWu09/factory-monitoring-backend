# Automatic Monitoring System

An intelligent, real-time monitoring backend system designed to enhance safety and operational efficiency in industrial environments. The system detects unauthorized personnel or vehicles through video surveillance and triggers automated alerts via a distributed backend pipeline.

## Features

- Real-time object detection using YOLO (You Only Look Once)
- Video stream analysis with OpenCV
- Asynchronous alert pipeline powered by Kafka
- Secure image storage via MinIO
- Custom backend logic in Python
- Database integration for alert logging and data management

## Tech Stack

- Language: Python  
- Computer Vision: YOLOv5, OpenCV  
- Messaging Queue: Apache Kafka  
- Storage: MinIO (S3-compatible)  
- Database: PostgreSQL / MongoDB (update based on your implementation)  
- Deployment: Docker (if used)

## System Architecture

Video Feed

│

▼

YOLO + OpenCV (Real-time CV processing)
│
▼
Kafka Producer (Alert Trigger)
│
▼
Kafka Consumer → MinIO + Database Logging
