from urllib.parse import urlparse, parse_qs
import re
import chardet
import string
import csv
import urlParsing
import pandas as pd

uriSchemes = r"C:\Users\ezele\Desktop\thesis\tdaPython\datasets\uriSchemes.txt"
phishTrainP= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\phishTrain.txt"
phishTestP= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\phishTest.txt"
notpTrainP= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\notpTrain.txt"
notpTestP= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\notpTest.txt"

def parse_url(url):
    parsed = urlparse(url)
    components = {
        "scheme": parsed.scheme,  # e.g., 'http', 'https'
        "domain": parsed.netloc,  # e.g., 'www.example.com'
        "path": parsed.path,      # e.g., '/path/to/resource'
        "params": parsed.params,  # e.g., parameters after ';'
        "query": parsed.query,    # e.g., query string after '?'
        "fragment": parsed.fragment,  # e.g., fragment after '#'
    }
    if parsed.query:
        components["query_params"] = parse_qs(parsed.query)
    return components


def splitFile(file_path):
    with open(file_path, 'rb') as raw_file:
        raw_data = raw_file.read()
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding']

    print(f"Detected encoding: {detected_encoding}")

    # Open the file using the detected encoding
    with open(file_path, 'r', encoding=detected_encoding) as file:
        lines = [line.strip() for line in file if line.strip()]

    return lines  # Return lines, not the file path


def extractURL2(lines):
    # Process the lines (not a file path here)
    urls = [line.strip() for line in lines if line.strip()]
    return list(set(urls))  # Remove duplicates and return


phishList = extractURL2(splitFile(phishTrainP))
notPhishList = extractURL2(splitFile(notpTrainP))
phishTestList = extractURL2(splitFile(phishTestP))
notPTestList = extractURL2(splitFile(notpTestP))

schemeList = extractURL2(uriSchemes)

#analyzing the uri, find #punctuations etc
def schemeCheck(url):
    if (urlparse(url).scheme) ==  'https':
        return 2
    elif (urlparse(url).scheme) == 'http':
        return 1
    else:
        return 0

def domainLength(url):
    return len(urlparse(url).netloc)

def digitsInDomain(url):
    dom = urlparse(url).netloc
    digit_count = sum(char.isdigit() for char in dom)
    return digit_count

def dotsInDomain(url):
    dom = urlparse(url).netloc
    dotCount = sum((char == ".") for char in dom)
    return dotCount

def punctInDomain(url):
    punctuation_count = 0
    dom = urlparse(url).netloc
    for char in dom:
        if char in string.punctuation: #!”#$%&'()*+,-./:;<=>?@[\]^_`{|}~
            punctuation_count += 1
    return punctuation_count

def pathLength(url):
    return len(urlparse(url).path)

def digitsInPath(url):
    path = urlparse(url).path
    digit_count = sum(char.isdigit() for char in path)
    return digit_count

def punctInPath(url):
    punctuation_count = 0
    dom = urlparse(url).path
    for char in dom:
        if char in string.punctuation: #!”#$%&'()*+,-./:;<=>?@[\]^_`{|}~
            punctuation_count += 1
    return punctuation_count
#counts how many lil segments separeted by .or -
#b is to signify if we are looking at the domain or the params+query+frag+
# b = true if domain
def segmentsOfDomain(url, b):
    if b:
        dom = urlparse(url).netloc
    else:
        dom = urlparse(url).params + urlparse(url).query + urlparse(url).fragment
    segments = [segment for segment in dom.replace('.', '-').split('-') if segment]
    lens = [(len(segment)/sum(len(st) for st in segments)) for segment in segments]
    return len(segments), ((sum(lens)/len(lens)) if not (len(lens)==0 ) else 0)

def checkSchemeQuery(url):
    dom = urlparse(url).params + urlparse(url).query + urlparse(url).fragment
    return any(s in dom for s in schemeList)


def phishWordAmount(cont_str, str_list):
    count = sum(cont_str.count(s) for s in str_list)
    return count
def containsIp(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    ipv4 = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    if re.search(ipv4, domain):
        return True
    return False
def analyze_url(url):
    # Get all the results from the functions
    scheme = schemeCheck(url)
    domainLen = domainLength(url)
    digitsInD = digitsInDomain(url)
    punctInD = punctInDomain(url)
    pathLen = pathLength(url)
    digitsInP = digitsInPath(url)
    punctInP = punctInPath(url)
    dotsD = dotsInDomain(url)
    segD =segmentsOfDomain(url)
    # Print the results in a nice format
    print(f"Analysis of URL: {url}")
    print(f"Scheme: {'https' if scheme == 2 else 'http' if scheme == 1 else 'None'}")
    print(f"Domain Length: {domainLen} characters")
    print(f"Digits in Domain: {digitsInD} characters")
    print(f"Punctuation Marks in Domain: {punctInD} characters")
    print(f"Path Length: {pathLen} characters")
    print(f"Digits in Path: {digitsInP} characters")
    print(f"Punctuation Marks in Path: {punctInP} characters")
    print(f"dotsInDomain : {dotsD} numberDomainSegments: {segD}")

#cols
columnNames = ["scheme", "domainLength", "digitsInDomain", "specialInDomain", "dotsInDomain", "numberDomainSegments",
           "lengthSegmentsRatioDomain","pathLength","digitsInPath", "specialInPath", "numberQuersSegments", "lengthQuerySegRatio"
            , "schemeInQuery","longestCharRatio", "longestDigitRatio", "phishWord", "ip",
           "phish"]



# word list from the word cloud
freqNew = ['blogspot', 'forum', 'blog', 'new', 'shop', 'google', 'ftp', 'ning', 'library', 'sport', 'maps google', 'mirror', 'business', 'store', 'repositorio', 'images google', 'lib', 'repository', 'china', 'dspace', 'doc', 'old', 'pixnet', 'spb', 'uni', 'wikipedia', 'archive', 'free', 'archive ubuntu', 'compute amazonaws', 'wordpress', 'dev', 'cocolog nifty', 'community', 'debian', 'time', 'book', 'home', 'medium', 'job', 'cse google', 'photo', 'help', 'art', 'pinterest', 'rlpdotca appspot', 'image', 'portal', 'france', 'life', 'fandom', 'auto', 'developer', 'club', 'hospital', 'eco', 'city', 'med', 'weebly', 'photoshelter', 'finance', 'gov', 'seesaa', 'line', 'game', 'test', 'center', 'radio', 'group', 'design', 'list', 'isolutions iso', 'medcoforum aemet', 'cht', 'video', 'history', 'edu', 'wikidot', 'research', 'pravda', 'net', 'pref', 'pro', 'data', 'academy', 'media', 'career', 'motor', 'garbusy home', 'fas harvard', 'm opera', 'ec2 ap', 'realty yandex', 'bibliotecadigital anvisa', 'japan', 'utm', 'market', 'univ', 'tour', 'comune', 'live', 'bip', 'trade', 'tistory', 'mobile', 'site', 'cmu', 'expert', 'digital', 'photography', 'staging', 'travel', 'hotel', 'music', 'journal', 'iccsys', 'opac', 'studio', 'kid', 'netlify', 'commerce', 'periodicos ufc', 'dot rlpdotca', 'ap southeast', 'southeast compute', 'archive canonical', 'search yahoo', 'finance walnutcreekguide', 'air nifty', 'msk', 'biblio', 'krakow', 'europa', 'sugtvrdjava', 'berlin', 'gitlab', 'beauty', 'mit', 'cdn']

def createCsv(uris, outputhP, phishOrNot, wordList):
    with open(outputhP, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columnNames)
        uris = list(set(uris))
        for sample in uris:
            longestCharSeq, longestDigSeq = urlParsing.longestContSeq(sample)
            domSeg, (domSegRatio) = segmentsOfDomain(sample, True)
            qSeg, qSegRatio = segmentsOfDomain(sample, False)
            #mathcCount, matchRatio = urlParsing.countMatchRatio(sample, wordList, .4)
            row = [ schemeCheck(sample), domainLength(sample), digitsInDomain(sample), punctInDomain(sample),
                   dotsInDomain(sample), domSeg, domSegRatio, pathLength(sample),
                   digitsInPath(sample), punctInPath(sample) ,qSeg, qSegRatio, 1 if (checkSchemeQuery(sample)) else 0,
                    longestCharSeq,longestDigSeq, phishWordAmount(sample, wordList),
                    (1 if containsIp(sample) else 0),
                   #mathcCount, matchRatio,
                   phishOrNot]
            writer.writerow(row)
    return pd.read_csv(outputhP)

def createBigCsv(pL, npL, path, wL):
    df1 = createCsv(pL, "big.csv", 1, wL)
    df2 = createCsv(npL, "big.csv", 0, wL)
    df =pd.concat([df1, df2], ignore_index=True)
    df.to_csv(path, index=False)
    return 0
pTrain= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\train.csv"
pTest = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\test.csv"


createBigCsv(phishList, notPhishList, pTrain, freqWordsP2)
createBigCsv(phishTestList, notPTestList, pTest, freqWordsP2)
