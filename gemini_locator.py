from API_KEY import GEMINI_KEY
from google import genai
import os

prompt = """
I will upload a photograph that is being used as a critical piece of evidence in a missing person investigation. Your task is to act as a Forensic Geospatial Intelligence Analyst to provide highly cautious, detailed, and actionable location leads.

Step 1: Specific Location Candidates (3-5 Primary Leads)
Based on the uploaded image, you must deduce and provide a list of 3 to 5 candidate locations (Primary Leads) that are as specific as possible (e.g., city, street intersection, specific address, or named landmark). You must provide 3-5 leads, even if one seems highly probable.

Forensic Visual Analysis Criteria (Focus on Specificity and Permanence):

Unique Built Environment: Specific, non-generic architectural details, utility pole markings, unique pavement patterns, or distinct public infrastructure (traffic signals, bollards, hydrants).

Hard Landmarks: Recognizable monuments, bridges, specific storefronts, distinct topographical features (e.g., unique mountain profiles).

Micro-Cues (Actionable Detail): Visible alphanumeric characters (license plates, building numbers, signs), unique trash receptacles, street furniture designs, or specific vehicle models/modifications common to a precise area.

Exclusionary Evidence: Briefly state what visual cues allow you to rule out other major regions or countries.

Output Format for Step 1 (Primary Leads):

For each of the 3-5 candidates, provide the following:

Location Lead: (Specific as possible: e.g., "Intersection of Main St. and 5th Ave., Metropolis, USA").

Analytic Confidence: (A percentage 0% to 100%) - This must reflect the probability of this being the exact location.

Detailed Reasoning & Key Evidence: An explanation detailing the most compelling, verifiable visual evidence that supports this lead. Example: "The unique yellow-diamond speed limit sign combined with the distinctive white-and-red curb paint strongly points to Italian municipal code."

Step 2: Broad Geographic Analysis (3-5 Secondary Leads)
If the most specific location leads (Step 1) all have an Analytic Confidence below 60%, your fallback task is to provide the best 3 to 5 probable countries or focused geographic regions (Secondary Leads). These must be more specific than a continent or an ocean.

Broad Analysis Criteria (Focus on Climate and Geology):

Climatic & Flora Markers: Dominant plant species, unique soil color/type, and indicators of specific climate zones (e.g., arid, temperate rainforest, alpine).

Geological & Terrain Types: General landforms, distinctive rock formations, and consistent patterns in horizon lines.

Shadow and Light Physics: Highly detailed analysis of sun angle, shadow length, and atmospheric conditions to estimate precise latitude, time of day, and season.

Output Format for Step 2 (Secondary Leads):

Region/Country Lead: (e.g., "The Iberian Peninsula (Spain/Portugal)" or "The Southeast US Piedmont Region").

Supporting Rationale: A concise explanation detailing the most reliable broad cues used.

Step 3: Risk Assessment and Uncertainty
You must conclude with a mandatory section addressing the limitations of the analysis.

Ambiguity Factors: State what specific elements in the image introduce the greatest uncertainty or could be misleading (e.g., generic mass-produced objects, lack of contrast, or poor resolution).

Source Reliability Score (1-5, where 5 is best): Assign a score indicating how reliable the visual evidence itself is for accurate geolocation (e.g., a high-res photo of a unique sign is a 5; a blurry photo of a generic tree is a 1).
"""
def gemini_finder(image_path):
    client = genai.Client(api_key= GEMINI_KEY)
    my_file = client.files.upload(file=image_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[my_file, prompt]
    )

    return response.text


    
def gemini_fileWriter(match_paths, target_person, output_dir = "ReportFindings"):
    
    output_file = os.path.join(output_dir, f"{target_person}_gemini_output.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for image_path in match_paths:

            f.write(f"# LLM output for missing person {target_person}")
            text_output = gemini_finder(image_path)

            f.write(f"## Input Image Path: {image_path}\n\n")
            f.write(text_output + "\n\n")
            f.write("-----\n\n")
            
    print(f"Saved combined Gemini output to {output_file}")
