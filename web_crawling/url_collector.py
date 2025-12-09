import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.devtools.v134.dom import get_attributes
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import requests
import json

links_list = ['https://www.thewhiskyexchange.com/c/40/single-malt-scotch-whisky',
              'https://www.thewhiskyexchange.com/c/304/blended-scotch-whisky',
              'https://www.thewhiskyexchange.com/c/309/blended-malt-scotch-whisky',
              'https://www.thewhiskyexchange.com/c/310/grain-scotch-whisky',
              'https://www.thewhiskyexchange.com/c/32/irish-whiskey',
              'https://www.thewhiskyexchange.com/c/35/japanese-whisky',
              'https://www.thewhiskyexchange.com/c/33/american-whiskey'
              ]

for link in links_list:
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('detach',True)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)...")

    driver = webdriver.Chrome(chrome_options)
    url_whisky_exchange = link
    driver.get(url_whisky_exchange)
    time.sleep(3)

    try:
        # 최대 5초까지 대기 후 'Accept' 버튼 클릭
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]"))
        ).click()
        print("쿠키 동의 처리 완료")
    except:
        print("쿠키 동의창 없음 또는 이미 처리됨")


    time.sleep(3)

    # 할인 팝업 (Shadow DOM) 닫기
    # try:
    #     shadow_host = WebDriverWait(driver, 10).until(
    #         EC.presence_of_element_located((By.CSS_SELECTOR, "div.overlay-container"))
    #     )
    #     shadow_root = driver.execute_script("return arguments[0].shadowRoot", shadow_host)
    #     close_btn = shadow_root.find_element(By.CSS_SELECTOR, "button.close-btn")
    #     driver.execute_script("arguments[0].click();", close_btn)
    #     print("✅ 할인 팝업 닫힘")
    # except Exception as e:
    #     print(f"❌ 실패: {e}")

    # 목록페이지 펼치기
    # 제품 로드 반복 16 번
    # 한 페이지 당 24 개씩 보여줌
    max_clicks = 4
    for i in range(max_clicks):
        try:
            show_more = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="content"]/section/div[2]/nav/a'))
            )
            driver.execute_script("arguments[0].scrollIntoView();", show_more)
            time.sleep(0.5)
            show_more.click()
            print(f"✅ Show More 클릭 {i+1}회")
            time.sleep(4)  # 로딩 대기

        except:
            print(f"❌ {i+1}번째 클릭 실패")
            break

    time.sleep(4)

    # 제품 상세페이지 링크 저장
    product_cards = driver.find_elements(By.CSS_SELECTOR, "a.product-card.js-product-view")

    product_link = [{'name' : p.get_attribute('title'), 'url': p.get_attribute('href')} for p in product_cards]

    category = link.rstrip('/').split('/')[-1]
    with open(f'links_data/{category}_links.json', 'w') as f:
        json.dump(product_link, f ,indent=3,ensure_ascii=False)

    print('done')
    driver.quit()
