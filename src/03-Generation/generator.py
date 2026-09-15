import os
import time
import random

from dotenv import load_dotenv
from google import genai

load_dotenv()


MODEL_NAME = "gemini-3.5-flash"

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")


client = genai.Client(api_key=API_KEY)


def generate_answer(question, context, max_retries=4):
    """
    Generate an answer using Gemini.

    Retries temporary Gemini API errors such as
    429, 500, 503 and 504 using exponential backoff.
    """

    prompt = f"""
You are a helpful knowledge assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I couldn't find enough reliable information in the knowledge base to answer this question."

Do not make up information.

Context:
{context}

User Question:
{question}

Answer clearly and concisely.
"""

    for attempt in range(max_retries + 1):

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:

            error_text = str(e).lower()

            # Errors that may be temporary
            is_retryable = any(
                error in error_text
                for error in [
                    "503",
                    "service unavailable",
                    "unavailable",
                    "429",
                    "resource exhausted",
                    "rate limit",
                    "too many requests",
                    "500",
                    "internal server error",
                    "504",
                    "timeout"
                ]
            )

            # Don't retry permanent errors
            if not is_retryable:
                raise e

            # Retries exhausted
            if attempt >= max_retries:
                raise e

            # Exponential backoff + random jitter
            delay = (2 ** attempt) + random.uniform(0, 1)

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {delay:.1f} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError(
        "Gemini failed after multiple retry attempts."
    )