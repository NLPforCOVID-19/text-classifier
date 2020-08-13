from newspaper import Article
import json
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--input", default='input.json', help="Path to annotation file")
argparser.add_argument("--output", default='output.json', help="Output file (JSONL)")
args = argparser.parse_args()



with open(args.input) as f:
    lists = f.readlines()
lists = [json.loads(x.strip()) for x in lists]
failed_list = []
exception = []
for sample in lists:
    print(sample)
    article = Article(sample['url'])
    try:
        article.download()
        article.parse()
        print(article.text)
        print(article.title)
        sample['title'] = article.title
        if article.text == u'':
            failed_list.append(sample)
            lists.remove(sample)
        else:
            sample['cleaned_text'] = article.text
    except:
        exception.append(sample)
        lists.remove(sample)
        continue
    # break


    # if article.cleaned_text == u'':
    #     raise Exception
    # input()
# print(lists[0])
lists = [json.dumps(x, ensure_ascii=False) for x in lists]
exception = [json.dumps(x, ensure_ascii=False) for x in exception]
failed_list = [json.dumps(x, ensure_ascii=False) for x in failed_list]
# print([isinstance(x, str) for x in lists])

with open(args.output, "w") as f:
    # f.writelines([json.dumps(x) for x in lists])
    for sample in lists:
        # print(sample)
        f.write(sample)
        f.write('\n')
with open('./exception.jsonl', "w") as f:
    for sample in exception:
        # print(sample)
        f.write(sample)
        f.write('\n')
with open('./failed.jsonl', "w") as f:
    for sample in failed_list:
        # print(sample)
        f.write(sample)
        f.write('\n')