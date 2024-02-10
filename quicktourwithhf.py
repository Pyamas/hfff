from transformers import pipeline

classifider = pipeline("sentiment-analysis")

classifider("We are very happy to show you the 🤗 Transformers library.")

results = classifider(["We are very happy to show you te transformers library."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
