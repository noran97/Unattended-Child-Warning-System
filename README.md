# 👶🚗 Child Safety Monitoring System with Azure Notifications  

This project implements an **AI-powered safety monitoring system** that detects **children left unattended in vehicles or environments**. It combines **face detection, age classification, and Azure cloud integration** to trigger real-time alerts when children are detected without adult supervision.  

---

## ✨ Features  

- 🧑‍🤝‍🧒 **Face Detection & Age Classification**  
  - Uses OpenCV DNN for face detection  
  - Classifies detected faces as **Child** or **Adult** using a custom-trained CNN model (VGG16 + NCNN module).  

- ⏱️ **Unattended Children Tracking**  
  - Triggers alerts if children are detected **without adults** for a configurable threshold (default: 60 seconds).  
  - Resets if adult supervision is restored or no children are detected.  

- ☁️ **Azure Notifications**  
  - Sends structured JSON alerts to **Azure Blob Storage**.  
  - Includes alert type, elapsed time, and resolution notifications.  
  - Cooldown system prevents spamming (default: 5 minutes).  

- 📹 **Video & Image Support**  
  - Works with **images**, **video files**, and **live webcam feeds**.  
  - Saves processed outputs with bounding boxes, labels, and warnings.  

- 🎛️ **Customizable Parameters**  
  - Adjustable image size, confidence thresholds, and warning times.  
  - Toggle Azure integration via URL.  



