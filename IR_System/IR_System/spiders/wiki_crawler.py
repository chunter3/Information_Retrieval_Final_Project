import scrapy
import requests

class WikiCrawler(scrapy.Spider):

    name = "wiki_crawler"

    allowed_domains = ['en.wikipedia.org']

    start_urls = ['https://en.wikipedia.org/wiki/Etrian_Odyssey']  

    def parse(self, response):

        # I've decided that only links w/in p or dl tags are relevant to crawl

        rel_p_links = [(href,title) for href,title in zip([href for href in response.css('p a::attr(href)').getall() if "cite_note" not in href], response.css('p a::attr(title)').getall())]

        rel_dl_links = [(href,title) for href,title in zip([href for href in response.css('dl a::attr(href)').getall() if "cite_note" not in href], response.css('dl a::attr(title)').getall())]  

        for href,title in [*rel_p_links, *rel_dl_links]:
            
            webpage_content = requests.get(f"http://en.wikipedia.org{href}")
             
            filename = f'{title}.html'
            html_files = [] 
            with open(filename, 'wb') as f:
                (f.write(webpage_content.content))

                html_files.append(f)

        f.close()

        seed_content = requests.get('https://en.wikipedia.org/wiki/Etrian_Odyssey')

        with open("Etrian_Odyssey.html", 'wb') as f:
            (f.write(seed_content.content))

            html_files.append(f)

        f.close()

        return html_files

