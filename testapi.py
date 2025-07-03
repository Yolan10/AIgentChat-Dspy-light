import os
import openai

if __name__ == "__main__":
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("No API key configured")
    else:
        try:
            client = openai.OpenAI(api_key=key)
            client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "ping"}])
            print("API key works")
        except Exception as e:
            print("API test failed", e)
