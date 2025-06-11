import time
import threading
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


class GoogleShoppingService:
    def __init__(self):
        """
        GoogleShoppingService 클래스의 인스턴스를 초기화하고 WebDriver를 설정합니다.
        드라이버는 한 번만 초기화됩니다.
        """
        self.lock = threading.Lock()
        # Chrome WebDriver를 자동으로 설정합니다.
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        # options.add_argument(
        #     "--headless"
        # )  # 브라우저 창을 띄우지 않고 실행하려면 이 주석을 해제하세요.
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

        self.driver = webdriver.Chrome(service=service, options=options)
        self.original_window = self.driver.current_window_handle

    def search_google_shopping(self, query: str):
        """
        Selenium을 사용하여 Google Shopping에서 상품 정보를 스크래핑합니다.
        요청이 들어오면 새 탭을 열어서 검색한 뒤 닫습니다.

        Args:
            query (str): 검색할 키워드.

        Returns:
            list: 상품명, 가격, 판매자가 포함된 딕셔너리의 리스트.
        """
        with self.lock:
            self.driver.switch_to.new_window("tab")

            search_query = query.replace(" ", "+")
            url = f"https://www.google.com/search?q={search_query}&tbm=shop"

            products = []
            try:
                self.driver.get(url)

                # 페이지가 로드되고 상품 목록이 나타날 때까지 최대 4초간 기다립니다.
                WebDriverWait(self.driver, 4).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "product-viewer-group")
                    )
                )

                # 각 상품 정보를 담고 있는 요소들을 모두 찾습니다.
                groups = self.driver.find_elements(
                    By.CSS_SELECTOR, "product-viewer-group g-card ul"
                )

                for group in groups:
                    items = group.find_elements(By.CSS_SELECTOR, "li")

                    for item in items:
                        try:
                            splited_item = item.text.split("\n")
                            image = item.find_element(
                                By.CSS_SELECTOR, "img"
                            ).get_attribute("src")

                            # slice item
                            if splited_item[0] == "세일" or splited_item[0] == "가격 인하":
                                name = splited_item[1]
                                price = splited_item[2]
                                seller = splited_item[3]
                            else:
                                name = splited_item[0]
                                price = splited_item[1]
                                seller = splited_item[2]

                            products.append(
                                {
                                    "image": image,
                                    "name": name,
                                    "price": price,
                                    "seller": seller,
                                }
                            )
                        except Exception as e:
                            # 일부 상품은 구조가 다르거나 정보가 없을 수 있으므로, 오류가 나도 계속 진행합니다.
                            print(f"하나의 상품 정보를 추출하는 데 실패했습니다: {e}")
                            continue

            except Exception as e:
                print(
                    "상품 목록을 로드하는 데 실패했습니다. 페이지 구조가 변경되었을 수 있습니다."
                )
                print(f"오류: {e}")
            finally:
                # 현재 탭을 닫습니다.
                self.driver.close()
                # 원래 탭으로 다시 전환합니다.
                self.driver.switch_to.window(self.original_window)

            return products

    def quit(self):
        """
        WebDriver를 종료합니다.
        """
        self.driver.quit()


if __name__ == "__main__":
    search_keyword = "iPhone X"

    if search_keyword:
        service = GoogleShoppingService()
        try:
            scraped_data = service.search_google_shopping(search_keyword)
        finally:
            service.quit()
