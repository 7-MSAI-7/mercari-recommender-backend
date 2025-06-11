import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def search_google_shopping(query: str):
    """
    Selenium을 사용하여 Google Shopping에서 상품 정보를 스크래핑합니다. (가장 안정적인 버전)

    Args:
        query (str): 검색할 키워드.

    Returns:
        list: 상품명, 가격, 링크, 판매자가 포함된 딕셔너리의 리스트.
    """
    # Chrome WebDriver를 자동으로 설정합니다.
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument(
        "--headless"
    )  # 브라우저 창을 띄우지 않고 실행하려면 이 주석을 해제하세요.
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # adding argument to disable the AutomationControlled flag
    options.add_argument("--disable-blink-features=AutomationControlled")

    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
    )

    # exclude the collection of enable-automation switches
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    # turn-off userAutomationExtension
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(service=service, options=options)

    search_query = query.replace(" ", "+")
    url = f"https://www.google.com/search?q={search_query}&tbm=shop"

    driver.get(url)

    # 페이지가 로드되고 상품 목록이 나타날 때까지 최대 10초간 기다립니다.
    # 'product-viewer-group'는 상품 목록 전체를 감싸는 컨테이너의 클래스입니다.
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "product-viewer-group"))
        )
    except Exception as e:
        print(
            "상품 목록을 로드하는 데 실패했습니다. 페이지 구조가 변경되었을 수 있습니다."
        )
        print(f"오류: {e}")
        driver.quit()
        return None

    # 각 상품 정보를 담고 있는 요소들을 모두 찾습니다.
    # 이 클래스는 개별 상품 카드를 나타냅니다.
    groups = driver.find_elements(By.CSS_SELECTOR, "product-viewer-group g-card ul")

    products = []
    for group in groups:
        items = group.find_elements(By.CSS_SELECTOR, "li g-inner-card")
        for item in items:
            try:
                # 상품명: 태그 안에 텍스트로 존재합니다.
                name = item.find_element(
                    By.CSS_SELECTOR, "div div div div div div div img"
                ).get_attribute("alt")
                print(f"name: {name}")

                # 가격: 태그와 특정 클래스 안에 있습니다.
                price = item.find_element(
                    By.CSS_SELECTOR, "div div div div div div div:nth-child(3)"
                ).text
                print(f"price: {price}")

                # 상품 링크: 상품 카드 전체를 감싸는 a 태그의 href 속성에서 가져옵니다.
                # link = item.find_element(By.CSS_SELECTOR, "").get_attribute('href')

                # 판매자: 태그와 특정 클래스 안에 있습니다.
                seller = item.find_element(
                    By.CSS_SELECTOR, "div div div div div div div:nth-child(4)"
                ).text
                print(f"seller: {seller}\n")

                products.append(
                    {
                        "name": name,
                        "price": price,
                        "seller": seller,
                    }
                )
            except Exception as e:
                # 일부 상품은 구조가 다르거나 정보가 없을 수 있으므로, 오류가 나도 계속 진행합니다.
                print(f"하나의 상품 정보를 추출하는 데 실패했습니다: {e}")
                continue

    driver.quit()
    return products


if __name__ == "__main__":
    search_keyword = "iPhone X"

    if search_keyword:
        scraped_data = search_google_shopping(search_keyword)
        if scraped_data:
            print(f"\n--- '{search_keyword}'에 대한 검색 결과 ---")
            for i, product in enumerate(scraped_data, 1):
                print(f"[{i}]")
                print(f"  상품명: {product['name']}")
                print(f"  가격: {product['price']}")
                print(f"  판매자: {product['seller']}")
                print("-" * 30)
        else:
            print("상품 정보를 가져오는 데 실패했습니다.")
