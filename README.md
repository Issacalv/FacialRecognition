# Facial Recognition 
This project is a python based facial recognition program inspired by the **"College Buddies"** side mission from *Marvel's Spider-Man*. In the mission, Spider-Man uses a photograph to locate a missing student. This project adapts that concept into a real-world application using computer vision and machine learning.

## 📘 Overview
This project demonstrates how to:
- Load and analyze reference images.
- Detect faces in images or live video feeds.
- Match faces against a target image using face embeddings.
- Provide feedback when a match is found.

## 🕵️ How It Works
### Step 1: Access the "Database"
- The system taps into a simulated database inspired by Spider-Man’s social media image-scraper.
- This database contains reference images, public metadata, and contextual clues.

### Step 2: Encode Faces
- The library converts the target face and database faces into numerical embeddings.

### Step 3: Compare Embeddings
- The program compares the target face to all database entries.
- If similarity passes a threshold, the system returns a match.

### Step 4: Handle Missing Metadata
- If the matched image lacks sufficient metadata (e.g., no location, no EXIF data), an LLM is used to infer likely locations based on:
  - Visual context
  - Landmarks
  - Environmental cues
  - Crowd patterns and objects in view

### Step 5: Output
Mission-themed output is shown:
```
[SCANNER ONLINE]
Searching for missing person...
Match found! Inferring location...
Location identified.
```
---