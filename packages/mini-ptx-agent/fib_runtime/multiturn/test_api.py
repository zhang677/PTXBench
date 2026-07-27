import os, requests, json

key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
model = "gemini-3.1-pro-preview"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

payload = {
"contents": [{"parts": [{"text": "Reply with exactly: ok"}]}],
"generationConfig": {
    "maxOutputTokens": 256,
    "temperature": 0
}
}

r = requests.post(url, params={"key": key}, json=payload, timeout=60)
print("status", r.status_code)

if r.ok:
    data = r.json()
    print("modelVersion", data.get("modelVersion"))
    print("finishReason", data.get("candidates", [{}])[0].get("finishReason"))
    print("parts", json.dumps(
        data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    )[:500])
else:
    print("body", r.text.replace("\n", " ")[:500])