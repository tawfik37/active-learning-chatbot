"""
Web Search Module
Uses Google Custom Search API to fetch ground truth answers
"""

from googleapiclient.discovery import build
from config import model_config as cfg


def get_web_answer(question):
    """Uses the Google Search API to get the 'ground truth' answer."""

    try:
        service = build("customsearch", "v1", developerKey=cfg.GOOGLE_API_KEY)
        result = service.cse().list(q=question, cx=cfg.GOOGLE_CSE_ID, num=3).execute()

        if 'items' not in result or not result['items']:
            print(" Google Search returned no results.")
            return None

        all_snippets = []
        for i, item in enumerate(result['items']):
            snippet = item['snippet'].replace("...", "").strip()
            all_snippets.append(f"Source {i+1}: {snippet}")

        mega_context = "\n".join(all_snippets)
        return mega_context

    except Exception as e:
        print(f"Error during Google Search: {e}")
        return None
