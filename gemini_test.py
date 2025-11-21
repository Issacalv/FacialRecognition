from API_KEY import GEMINI_KEY
from google import genai

client = genai.Client(api_key= GEMINI_KEY)

my_file = client.files.upload(file="")

prompt = """
I will upload a photograph and your job is to determine where the location on the image based on:
- Visual Context
- Landmarks
- Environmental or Background Cues
- Crowd Patterns and objects in view
- Clothing cues

You will provide a list of 3-5 location candidates based on the prior information. (Even if you are 100 percent confident on a location)
Confidence scores for each location and with reasoning.

If a specific location cannot be determined, provide the best 3-5 cities or countries based on
- Geographic visuals
- Architecture
- Natural vegetation
- Terrain types
- Atmospheric Cues
- Shadow Physics

If uncertain of any location, state what makes it difficult to determine.


"""
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[my_file, prompt]
)

print(response.text)