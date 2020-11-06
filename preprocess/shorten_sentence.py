import re
import unicodedata


COVID_PATTERNS = [re.compile(r'新型コロナウ[イィ]ルス(感染症|肺炎)?'),
                  re.compile(r'COVID[-−]19感染症')]
DATE_PATTERN = re.compile(r'2020年([1-9]|1[0-2])月([1-9]|1[0-9]|2[0-9]|3[0-1])日')
SOURCES = [re.compile(r'[-−] Yahoo! JAPAN'),
           re.compile(r'[-−] Yahoo!ニュース'),
           re.compile(r'[-−]((The )?New York Times|ニューヨークタイムズ)')]
TIME_PATTERNS = [re.compile(r'[<\(\[\{]第.+?回[>\)\]\}]'),
                 re.compile(r'第.+?回'),
                 re.compile(r'[<\(\[\{]令和.+?年度[>\)\]\}]'),
                 re.compile(r'令和.+?年度')]


def shorten(sent, is_title=True):
    sent = unicodedata.normalize("NFKC", sent)   # 全角->半角(正規化)
    if is_title:
        sent = sent.split('_')[0].split('|')[0]  # 特定の記号で繋がれるsuffixは除く
    sent = sent.replace('中華人民共和国', '中国')
    sent = re.sub(DATE_PATTERN, r'\1月\2日', sent)
    for pat in TIME_PATTERNS + SOURCES:
        sent = re.sub(pat, '', sent)
    for pat in COVID_PATTERNS:
        sent = re.sub(pat, 'コロナ', sent)
    if is_title and len(sent) >= 10 and sent.endswith('について'):
        sent = sent.rstrip('について')
    return sent.strip()
