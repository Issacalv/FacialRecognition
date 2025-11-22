
# 🕷️ Facial Recognition + LLM Photo Analysis  
A real-world reinterpretation of the **"College Buddies"** mission from *Marvel’s Spider-Man* — where Spider-Man uses a single photo to locate a missing student.  
This project builds a working version of that idea using **computer vision**, **face embeddings**, **image databases**, and **LLM-powered environment inference**.
This is assuming that *no* metadata is available to extract.

---

# 📘 Overview
This system takes a **target image** of a person and:

1. Encodes the target face.
2. Scans a directory of images (your "database").
3. Finds the closest facial matches using Euclidean distance.
4. Selects the **top N most similar images**.
5. Sends those images to a large language model (LLM) to infer:
   - visual context  
   - objects  
   - possible locations  
   - surrounding environment  

Finally, it generates a **single consolidated markdown/text report** containing all LLM results.

---

# 🔍 How It Works (Step-by-Step)

## **1. Image Database Scanning**
The system walks through a folder (e.g., `Dataset/`) and extracts:
- image filename  
- full file path  
- number of faces  
- face encodings  

Encodings are cached for speed.

---

## **2. Target Face Encoding**
The target image (e.g., `"Person1.jpg"`) is converted into a 128-D facial embedding using the `face_recognition` library.

This numeric representation allows for distance-based comparison.

---

## **3. Face Matching + Distance Ranking**
For every encoded face in the database:

- The system compares it to the target face.
- Computes a **distance score**.
- Filters out poor matches.
- Sorts by similarity.

A table of all matches is saved to:

```
/ReportFindings/<target_person>.csv
```

The **top N** closest images are selected automatically.

---

## **4. LLM Context Analysis**
Each of the top N matched images is sent into an LLM (Gemini 2.5 or later) along with a prompt such as:

> “Analyze this image for environmental context, landmarks, or clues that may indicate location or surroundings.”

The model returns:
- descriptions  
- contextual clues  
- inferred location hints  
- notable visual elements  

This helps replicate the idea of “Spider-Man analyzing surroundings to find someone.”

---

## **5. Unified Report Generation**
All LLM responses are combined into **one output file**:

```
/GeminiOutput/<target_name>_gemini_output.txt
```

For each matched image, the report includes:
- A heading with the image path  
- A subheading placeholder  
- The LLM-generated analysis  
- A separator  

Example:

```
# Input Image Path: Dataset/Image14.jpg

<LLM Output>

-----
```

This acts like a digital case file.

---

# 🚀 Features
- Facial detection and encoding  
- Database-wide batch comparison  
- Automatic embedding caching  
- Distance-based ranking of matches  
- LLM contextual image analysis  
- Clean, single-file report combining all results  
- Spider-Man–inspired theming  

---

# 🧱 Technologies Used
- **Python 3**
- `face_recognition`
- `numpy`, `pandas`
- Gemini / Google GenAI LLM API
---

# 📂 Output Structure

```
/ReportFindings
    Person1.csv
    Person1_gemini.output.txt

/Dataset
    <images>

/DatabaseEncodings
    <cached .npy files>
```

---

# 🧭 Future Enhancements
- Geolocation metadata extraction  
- Multi-person comparison  
- Scene/object-weighted scoring  
