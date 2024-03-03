import requests
from bs4 import BeautifulSoup
from datasets import load_dataset

def crawl_wikipedia_page(name):
    # 构建Wikipedia页面的URL
    url = f'https://en.wikipedia.org/wiki/{name}'

    # 发送HTTP请求获取页面内容
    response = requests.get(url)

    if response.status_code == 200:
        # 使用BeautifulSoup解析页面内容
        soup = BeautifulSoup(response.text, 'html.parser')

        # 提取页面上的所有段落文本
        paragraphs = soup.find_all('p')

        # 将所有段落文本合并成一个字符串
        all_text = '\n'.join([paragraph.text for paragraph in paragraphs])

        return all_text
    else:
        raise Exception(f'Failed to fetch Wikipedia page for {name}')


if __name__ == '__main__':
    dataset = load_dataset("squad")
    wiki_titles = []
    for key in dataset.keys():
        for sample in dataset[key]:
            wiki_titles.append(sample['title'])
    wiki_titles = list(set(wiki_titles))
    print(f"==> Found {len(wiki_titles)} unique Wikipedia titles")
    for title in wiki_titles:
        try:
            wiki_content = crawl_wikipedia_page(title)
        except:
            continue
        with open(f"../database/{title}.txt", "w") as f:
            f.write(wiki_content)