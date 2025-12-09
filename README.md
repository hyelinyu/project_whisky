## 📁 Project Structure

```text
whisky_sales_strategy/
├── 02_data_preprocessing.ipynb
├── 03_exploratory_analysis.ipynb
├── 04_flavour_based_framework.ipynb
├── 05_recommendation_system.ipynb
├── 06_evaluation_metrics.ipynb
│
├── dataset/  
│   ├── whisky_data.csv
│   ├── whisky_framed.csv
│   ├── whisky_processed.csv
│   └── whisky_recommendation.csv
│
├── model/
│   ├── recommender_model.py
│   └── whisky_recommender.pkl
│
└── web_crawling/
    ├── url_collector.py
    └── 01_data_collector.py
```

#### Dataset

```
The original dataset used in this project is **not included in this repository**  
to respect the content ownership and usage policies of the source website  
(The Whisky Exchange).

Only the **analysis code** and **modeling pipeline** are shared here.
If you wish to reproduce the results, please collect the data directly from the
original source according to their Terms & Conditions.

본 프로젝트에서 사용된 원본 데이터셋은 소스 웹사이트
(The Whisky Exchange)의 콘텐츠 소유권 및 이용 정책을 준수하기 위해
이 저장소에 포함하지 않았습니다.

이 저장소에는 **분석 코드**, **파생 변수**, **모델링 파이프라인**만 공유되어 있습니다.
결과를 재현하고자 하는 경우, 해당 웹사이트의 이용 약관에 따라
원본 데이터를 직접 수집하여 사용하시기 바랍니다.
```
dataset resource : <a href="[https://www.thewhiskyexchange.com/](https://www.thewhiskyexchange.com/)" target="_blank">The Whisky Exchange</a>
