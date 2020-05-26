import json
article_dict = {}
with open("data/output.json", encoding="utf-8") as f:
    articles = f.readlines()
    articles = [json.loads(item) for item in articles]
for item in articles:
    article_dict[item["meta"]["url"]] = item
with open("data/crowdsourcing20200420.jsonl") as f:
    annotations = f.readlines()
    annotations = [json.loads(item) for item in annotations]

deletion = 0
for anno in annotations:
    if anno["url"] in article_dict.keys():
        anno["text"] = article_dict[anno["url"]]["text"]

annotations = [item for item in annotations if "text" in item.keys()]
# print(deletion)
print(len(annotations))
# print(annotations)
# print(annotations)


with open("data/crowdsourcing20200420.processed.jsonl", "w") as f:
    for item in annotations:
        f.write(json.dumps(item))
        f.write("\n")



# print(articles)
