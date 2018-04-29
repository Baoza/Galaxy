import os
import pandas as pd
import bs4
import requests
import numpy as np

# STEP 1
# re-write .cvs to .txt

table = pd.read_csv("summary_yinhan.csv", encoding="utf-8")
id_txt_name = './id.txt'
url_txt_name = './url.txt'

with open(id_txt_name, 'w') as outfile:
    table["objid"].to_string(outfile)


# STEP 2
# find downloading-url for one spectra
# use dbo.fGetUrlFitsSpectrum

def query_site(site):
    """
    :param site: website url
    :return: soup object for site
    """
    try:
        web = requests.get(site)
    except requests.exceptions.RequestException as e:
        raise ValueError('page not found')

    soup = bs4.BeautifulSoup(web.text, 'lxml')
    return soup


def Get_FITS_URL(specid):
    bound = []
    SQL = "http://skyserver.sdss.org/dr14/en/tools/search/x_results.aspx?" \
          "searchtool=SQL&TaskName=Skyserver.Search.SQL&syntax=NoSyntax&ReturnHtml" \
          "=true&cmd=select+dbo.fGetUrlFitsSpectrum%28"+str(specid)+"%29+&format=html&TableName="

    soup = query_site(SQL)
    td = soup.body.find_all('td')
    if len(td) != 2:
        return None
    else:
        return td[1].text
    return None

# write the URLs into .txt
batch_size = 1000
rm_list = []
k = 0

id_list = open(id_txt_name, 'r')
with open(url_txt_name, 'w') as outfile:
    for id in id_list:
        url = Get_FITS_URL(id)
        outfile.write(url)

# install gsplit by using "brew install coreutils"
os.system("gsplit -l " + str(batch_size) + "--additional-suffix=.txt $url.txt file")


def processing():
    return 0

for i in range(10):
    # STEP 3: download FITS
    os.system("wget -i /Users/honka/Desktop/galaxy/"
              "x0" + str(i) + "-P /Users/honka/desktop/galaxy/data")
    os.system("sudo rm -rf ./data")

    processing()

"""
total_size = 0
for i in np.arange(11, total_size):
    os.system("wget -i /Users/honka/Desktop/galaxy/"
              "x0" + str(i) + "-P /Users/honka/desktop/galaxy/data")
    os.system("sudo rm -rf ./data")

"""

