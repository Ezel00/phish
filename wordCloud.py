from urllib.parse import urlparse
import matplotlib.pyplot as plt
from tldextract import tldextract
from wordcloud import WordCloud

def readF(file_path):
    with open(file_path,  'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]
      
#gets rid of the bits not needed for the phishing url keywords
def discardBits(l):
    def removeTld(url):
        extracted = tldextract.extract(url)
        return f"{extracted.subdomain}.{extracted.domain}" if extracted.subdomain else extracted.domain

    def removeScheme(url):
        parsed = urlparse(url)
        netloc = parsed.netloc.lstrip("www.")
        return netloc + parsed.path

    o = [removeScheme(url) for url in l]
    o2 = [removeTld(url) for url in o]
    return o2


phishes = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\phishTrain.txt"
notPhishes = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\notpTrain.txt"
pList = readF(phishes)
npList = readF(notPhishes)

def wordCloud(inpList, p):
    notPhishData = " ".join(inpList)
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(notPhishData)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  # Hide axis
    plt.title("Not Phish Word Cloud" if not p else "Phish Word Cloud")
    plt.show()
    word_list = list(wordcloud.words_.keys())
    return word_list

phishL =(wordCloud(discardBits(pList), 1))
notPL = (wordCloud(discardBits(npList), 0))

def discardCommons(l1, l2):
    l1 = [item for item in l1 if (item not in l2) and (len(item))>2]
    return l1

#is the final list of keywords found in phishing urls and not in legitimate urls
l= discardCommons(phishL, notPL)
