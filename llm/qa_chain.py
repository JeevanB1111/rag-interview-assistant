import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.

Answer the question ONLY using the context provided below.
If the answer is not found in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)


    return response.text
