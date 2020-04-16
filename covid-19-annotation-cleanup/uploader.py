from doccano_api_client import DoccanoClient

# instantiate a client and log in to a Doccano instance
doccano_client = DoccanoClient(
    'http://54.199.70.161/',
    'christopher',
    'cjj8120972'
)

with open("data/uploadlist") as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        line = line.rstrip('\n')
        r_json_upload = doccano_client.post_doc_upload(4, 'json', line[2:], 'data/tbupload')
