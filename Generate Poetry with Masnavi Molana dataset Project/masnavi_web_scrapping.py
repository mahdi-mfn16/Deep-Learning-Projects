# scrapping web page

from lxml import html
import requests

for i in range(1,7):  #for 6 daftar of masnavi
	page = requests.get('https://ganjoor.net/moulavi/masnavi/daftar' + str(i) + '/')  #page with poetry links
	tree = html.fromstring(page.content)
	poetry_lists = tree.xpath('//article[@id="garticle"]/p/a/@href')

	# print(poetry_lists)
	for j in range(len(poetry_lists)):
        page2 = requests.get('https://ganjoor.net' + poetry_lists[j])
        tree2 = html.fromstring(page2.content)
        mesraas = tree2.xpath('//article[@id="garticle"]/div[@class="b"]/div/p/text()')
        for mesra in mesraas:
            with open('masnavi' , 'a' , encoding="utf-8") as f: #file_name
                f.write('| ' + str(mesra) + '\n')



