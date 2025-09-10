import json
import time

import requests as rq
from bs4 import BeautifulSoup
import re
from unidecode import unidecode

class Scraping:

    def __init__(self):
        self.url_scraping = "https://www.chavesnamao.com.br"
        self.url_neighborhood = "https://www.chavesnamao.com.br/api/realestate/aggregations/navigationFilters/?viewport=desktop&level1=casas-a-venda&level2=sp-campinas&limit=470"
        self.list_neighborhood_find = []

    def get_neighborhood(self):
        response = rq.get(url=f"{self.url_neighborhood}")
        json_response = response.json()
        self.list_neighborhood_find = [item["url"] for item in json_response["data"].get("items", [])]
        return self.list_neighborhood_find


    def get_information(self):
        houses = []
        self.get_neighborhood()
        for neighborhood_url in self.list_neighborhood_find:
            for i in range(1, 5):
                try:
                    response_scraping = rq.get(url=f"{self.url_scraping}{neighborhood_url}/?pg={i}")
                    soup = BeautifulSoup(response_scraping.text, "html.parser")
                    cards = soup.find_all("div", {"data-template": "list"})

                    for card in cards:
                        price_tag = card.find("p", {"aria-label": "Preço"})
                        price = None
                        if price_tag:
                            price_text = re.split(r"Iptu|Condomínio", price_tag.get_text(strip=True))[0]
                            price_only = re.sub(r"[^\d]", "", price_text)
                            if price_only:
                                price = int(price_only)

                        address_tag = card.find("address")
                        neighborhood = None
                        if address_tag:
                            p_tags = address_tag.find_all("p")
                            if len(p_tags) >= 2:
                                neighborhood_raw = p_tags[1].get_text(strip=True).split(",")[0]
                                neighborhood = unidecode(neighborhood_raw.strip().lower())

                        area_tag = card.find("p", title=lambda t: t and "Área útil" in t)
                        area = None
                        if area_tag:
                            area_text = area_tag.get_text(strip=True)
                            area_num = re.sub(r"[^\d]", "", area_text)
                            if area_num:
                                area = int(area_num)

                        bedroom_tag = card.find("p", title=lambda t: t and "Quartos" in t)
                        bedroom = None
                        if bedroom_tag:
                            bedroom_num = re.sub(r"[^\d]", "", bedroom_tag.get_text(strip=True))
                            if bedroom_num:
                                bedroom = int(bedroom_num)

                        bathroom_tag = card.find("p", title=lambda t: t and "Banheiro" in t)
                        bathroom = None
                        if bathroom_tag:
                            bathroom_num = re.sub(r"[^\d]", "", bathroom_tag.get_text(strip=True))
                            if bathroom_num:
                                bathroom = int(bathroom_num)

                        garage_tag = card.find("p", title=lambda t: t and "Garagens" in t)
                        garage = None
                        if garage_tag:
                            garage_num = re.sub(r"[^\d]", "", garage_tag.get_text(strip=True))
                            if garage_num:
                                garage = int(garage_num)

                        if all([price, neighborhood, area, bedroom, bathroom, garage]):
                            houses.append({
                                "price": price,
                                "neighborhood": neighborhood,
                                "area": area,
                                "bedroom": bedroom,
                                "bathroom": bathroom,
                                "garage": garage
                            })
                except rq.exceptions.RequestException as e:
                    time.sleep(2)
        return houses
