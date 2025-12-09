import json
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import os

# Helper function for safe text extraction
def safe_get_text(soup_obj, tag_name, class_name):
    element = soup_obj.find(name=tag_name, class_=class_name)
    return element.getText(strip=True) if element else np.nan

# 중간 저장 경로
PROGRESS_LOG = 'scraped_urls.txt'
DATA_SAVE_PATH = 'whisky_data_last.csv'

# 중간 저장된 URL 불러오기
if os.path.exists(PROGRESS_LOG):
    with open(PROGRESS_LOG, 'r') as f:
        already_scraped = set(line.strip() for line in f)
else:
    already_scraped = set()

json_name = os.listdir('./links_data')
for file in json_name:
    with open(f'links_data/{file}', 'r') as f:
        data = json.load(f)

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('detach', True)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)...")

    for whisky in data:
        url = whisky['url']
        if url in already_scraped:
            print(f"이미 수집된 URL: {url}")
            continue

        driver = webdriver.Chrome(chrome_options)
        driver.get(url)
        time.sleep(5)

        try:
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]"))
            ).click()
            print("쿠키 동의 처리 완료")
        except:
            print("쿠키 동의창 없음 또는 이미 처리됨")

        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Main Info
        product_name = safe_get_text(soup, 'h1', 'product-main__name')
        product_main_info = safe_get_text(soup, 'ul', 'product-main__meta')
        vol_alcohol = safe_get_text(soup, 'p', 'product-main__data')
        product_price = safe_get_text(soup, 'p', 'product-action__price')

        time.sleep(3)

        # Style Info
        style_dict = {'style_body': np.nan, 'style_richness': np.nan, 'style_smoke': np.nan, 'style_sweetness': np.nan}
        try:
            style_tags = soup.find_all(name='div', class_='flavour-profile__gauge js-flavour-profile__gauge gauge-container')
            values = [tag.getText(strip=True) for tag in style_tags]
            for key, val in zip(style_dict.keys(), values):
                style_dict[key] = val
        except Exception as e:
            print('style parsing error:', e)

        time.sleep(3)

        # Characters
        characters = []
        try:
            char_tags = soup.find_all(name='li', class_='flavour-profile__item flavour-profile__item--character')
            characters = [tag.getText(strip=True) for tag in char_tags] if char_tags else [np.nan]
        except Exception as e:
            print('character parsing error:', e)

        time.sleep(2)

        # Food Pairing
        paring_dishes = []
        try:
            food_tags = soup.find_all(name='li', class_='food-pairing__item')
            paring_dishes = [tag.getText(strip=True) for tag in food_tags] if food_tags else [np.nan]
        except Exception as e:
            print('Food paring parsing error:', e)

        time.sleep(3)

        # Product Facts
        product_facts_dict = {'bottler': np.nan, 'age': np.nan, 'country': np.nan, 'region': np.nan, 'cask type': np.nan, 'colouring': np.nan,'vintage':np.nan}
        try:
            fact_types = soup.find_all(name='h3', class_='product-facts__type')
            fact_data = soup.find_all(name='p', class_='product-facts__data')
            for type_tag, data_tag in zip(fact_types, fact_data):
                key = type_tag.getText(strip=True).lower()
                value = data_tag.getText(strip=True).lower()
                if key in product_facts_dict:
                    product_facts_dict[key] = value
        except Exception as e:
            print('product facts parsing error:', e)

        time.sleep(3)

        # Tasting Notes
        tasting_notes_dict = {'nose': np.nan, 'palate': np.nan, 'finish': np.nan}
        try:
            note_titles = soup.find_all(name='h4', class_='product-notes__title')
            note_contents = soup.find_all(name='p', class_='product-notes__copy')
            for title_tag, content_tag in zip(note_titles, note_contents):
                key = title_tag.getText(strip=True).lower()
                value = content_tag.getText(strip=True)
                if key in tasting_notes_dict:
                    tasting_notes_dict[key] = value
        except Exception as e:
            print('tasting notes parsing error:', e)

        time.sleep(3)

        # Reviews
        reviews = []
        try:
            show_more = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.cta-button.cta-button--delta.js-review-list__more.review-list__more'))
            )
            driver.execute_script("arguments[0].scrollIntoView();", show_more)
            time.sleep(0.5)
            show_more.click()
            time.sleep(1)
        except:
            pass

        time.sleep(7)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        review_tags = soup.find_all(name='p', class_='review-list__copy')
        reviews = [tag.getText(strip=True) for tag in review_tags if 'service' not in tag.getText(strip=True).lower() and 'deliver' not in tag.getText(strip=True).lower()]
        reviews = [str(r) for r in reviews if pd.notnull(r)]
        if not reviews:
            reviews = [np.nan]

        # Row Construction
        row = {
            'name': product_name,
            'main_info': product_main_info,
            'volume_alcohol': vol_alcohol,
            'price': product_price,
            'style_body': style_dict['style_body'],
            'style_richness': style_dict['style_richness'],
            'style_smoke': style_dict['style_smoke'],
            'style_sweetness': style_dict['style_sweetness'],
            'characters': ', '.join([str(c) for c in characters if pd.notnull(c)]),
            'food_paring': ', '.join([str(d) for d in paring_dishes if pd.notnull(d)]),
            'bottler': product_facts_dict['bottler'],
            'age': product_facts_dict['age'],
            'country': product_facts_dict['country'],
            'region': product_facts_dict['region'],
            'casktype': product_facts_dict['cask type'],
            'colouring': product_facts_dict['colouring'],
            'vintage': product_facts_dict['vintage'],
            'nose': tasting_notes_dict['nose'],
            'palate': tasting_notes_dict['palate'],
            'finish': tasting_notes_dict['finish'],
            'reviews': ' || '.join([str(r) for r in reviews]),
            'url': url
        }

        # Append to CSV immediately
        df_temp = pd.DataFrame([row])
        df_temp.to_csv(DATA_SAVE_PATH, mode='a', header=not os.path.exists(DATA_SAVE_PATH), index=False)

        # Save progress
        with open(PROGRESS_LOG, 'a') as f:
            f.write(url + '\n')

        print(f'completed : scrapping for {url}')
        driver.quit()
        time.sleep(3)