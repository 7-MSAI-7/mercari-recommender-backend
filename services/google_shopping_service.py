import asyncio
from playwright.async_api import async_playwright
import re
import os
import time
from contextlib import asynccontextmanager
import random


# --- Configuration ---
# 환경 변수 'PLAYWRIGHT_POOL_SIZE'에서 풀 크기를 읽어옵니다.
# 설정되어 있지 않으면 기본값으로 8을 사용합니다.
DEFAULT_POOL_SIZE = 8
POOL_SIZE = int(os.environ.get("PLAYWRIGHT_POOL_SIZE", DEFAULT_POOL_SIZE))


class PlaywrightPagePool:
    """
    Playwright Page 인스턴스를 관리하는 비동기 풀입니다.
    미리 정해진 수의 Page 인스턴스를 생성하고 요청 시마다 대여/반납하여
    동시 스크래핑 요청을 효율적으로 처리합니다.
    """

    def __init__(self, pool_size=POOL_SIZE):
        """
        풀을 초기화합니다. 실제 브라우저와 페이지 생성은 initialize() 메서드에서 비동기적으로 수행됩니다.
        Args:
            pool_size (int): 풀에서 유지할 페이지의 최대 개수.
        """
        if pool_size <= 0:
            raise ValueError("풀 크기는 0보다 큰 정수여야 합니다.")
        self._pool_size = pool_size
        self._pages = asyncio.Queue(maxsize=pool_size)
        self._playwright = None
        self._browser = None

    async def initialize(self):
        """
        Playwright를 시작하고 브라우저와 페이지 풀을 생성합니다.
        """
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-web-security",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        for _ in range(self._pool_size):
            page = await self._create_page()
            await self._pages.put(page)

    async def _create_page(self):
        """
        탐지를 회피하고 불필요한 리소스를 차단하는 설정이 적용된
        새로운 Page 인스턴스를 생성합니다.
        """
        context = await self._browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            timezone_id="Asia/Seoul",
            locale="ko-KR",
        )

        async def block_unwanted_resources(route):
            if route.request.resource_type in ("stylesheet", "font", "image"):
                await route.abort()
            else:
                await route.continue_()

        await context.route(re.compile(r".*"), block_unwanted_resources)
        page = await context.new_page()
        return page

    @asynccontextmanager
    async def get_page(self):
        """
        풀에서 페이지를 가져오기 위한 비동기 컨텍스트 관리자를 제공합니다.
        'async with' 구문과 함께 사용하면 사용 후 페이지가 자동으로 풀에 반환됩니다.
        """
        page = await self._pages.get()
        try:
            yield page
        finally:
            try:
                await page.goto("about:blank")
                await self._pages.put(page)
            except Exception as e:
                print(f"페이지 리셋 중 오류 발생, 새 페이지로 교체합니다: {e}")
                try:
                    await page.context.close()
                except Exception as close_e:
                    print(f"기존 컨텍스트 종료 중 오류: {close_e}")
                new_page = await self._create_page()
                await self._pages.put(new_page)

    async def shutdown(self):
        """
        풀에 있는 모든 페이지와 컨텍스트를 닫고, 브라우저를 종료합니다.
        """
        print("Shutting down Playwright browser...")
        while not self._pages.empty():
            page = await self._pages.get()
            try:
                await page.context.close()
            except Exception as e:
                print(f"페이지/컨텍스트 종료 중 오류 발생: {e}")

        if self._browser and self._browser.is_connected():
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        print("Playwright shut down.")


class GoogleShoppingService:
    """
    PlaywrightPagePool을 사용하여 Google Shopping 스크래핑을 수행하는 서비스 클래스입니다.
    """

    def __init__(self, pool_size=POOL_SIZE):
        self.pool = PlaywrightPagePool(pool_size=pool_size)

    async def initialize(self):
        await self.pool.initialize()

    async def search(self, query: str):
        """
        PlaywrightPagePool에서 페이지를 받아 스크래핑을 수행합니다.
        실패 시 재시도 로직과 CAPTCHA 감지 기능을 포함합니다.
        """
        MAX_RETRIES = 2
        for attempt in range(MAX_RETRIES):
            try:
                products = []
                # get_page()가 컨텍스트 관리자이므로, 루프 안에서 매번 호출해야 합니다.
                async with self.pool.get_page() as page:
                    search_query = query.replace(" ", "+")
                    url = f"https://www.google.com/search?q={search_query}&tbm=shop"
                    await page.goto(url, wait_until="domcontentloaded", timeout=15000)

                    # CAPTCHA 또는 블락 페이지 감지
                    if "sorry/index" in page.url:
                        # 재시도를 위해 의도적으로 예외 발생
                        raise Exception(f"CAPTCHA page detected for query: {query}")

                    # 상품 컨테이너가 로드될 때까지 대기
                    await page.wait_for_selector("product-viewer-group", timeout=10000)
                    groups = await page.query_selector_all(
                        "product-viewer-group g-card ul"
                    )

                    # 상품이 없는 경우도 성공으로 간주
                    if not groups:
                        return []

                    for group in groups:
                        items = await group.query_selector_all("li")
                        for item in items:
                            try:
                                full_text = await item.inner_text()
                                splited_item = full_text.split("\n")
                                image_element = await item.query_selector("img")
                                image = (
                                    await image_element.get_attribute("src")
                                    if image_element and "data:image" in await image_element.get_attribute("src")
                                    else "No Image"
                                )
                                if (
                                    splited_item[0] in ("세일", "가격 인하")
                                    and len(splited_item) > 3
                                ):
                                    name, price, seller = (
                                        splited_item[1],
                                        splited_item[2],
                                        splited_item[3],
                                    )
                                elif len(splited_item) > 2:
                                    name, price, seller = (
                                        splited_item[0],
                                        splited_item[1],
                                        splited_item[2],
                                    )
                                else:
                                    continue
                                products.append(
                                    {
                                        "image": image,
                                        "name": name,
                                        "price": price,
                                        "seller": seller,
                                    }
                                )
                            except IndexError:
                                print(
                                    f"상품 정보 파싱 중 데이터가 부족하여 건너뜁니다: {full_text}"
                                )
                                continue
                            except Exception as e:
                                print(f"하나의 상품 정보를 추출하는 데 실패했습니다: {e}")
                                continue

                    # 성공적으로 상품을 파싱했으면 결과 반환
                    return products

            except Exception as e:
                print(
                    f"스크래핑 시도 {attempt + 1}/{MAX_RETRIES} 실패 (쿼리: {query}). 오류: {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    # 다음 시도 전 랜덤 딜레이
                    await asyncio.sleep(random.uniform(1, 3))
                else:
                    # 최종 실패
                    print(f"최종 스크래핑 실패 (쿼리: {query}).")
                    return []
        return []  # 루프가 모두 실패한 경우

    async def close(self):
        await self.pool.shutdown()


# --- 싱글톤 서비스 인스턴스 관리 ---
_google_shopping_service_instance = None

async def get_google_shopping_service():
    global _google_shopping_service_instance
    if _google_shopping_service_instance is None:
        _google_shopping_service_instance = GoogleShoppingService(pool_size=POOL_SIZE)
        await _google_shopping_service_instance.initialize()
    return _google_shopping_service_instance

async def close_google_shopping_service():
    global _google_shopping_service_instance
    if _google_shopping_service_instance:
        await _google_shopping_service_instance.close()
        _google_shopping_service_instance = None

async def search_google_shopping(query: str):
    service = await get_google_shopping_service()
    return await service.search(query)


# --- 동시 실행 데모 ---
async def worker(search_term, worker_id):
    """
    개별 비동기 워커에서 스크래핑을 수행합니다.
    """
    print(f"[워커 {worker_id}] 시작: '{search_term}' 검색 중...")
    start_time = time.time()
    results = await search_google_shopping(search_term)
    end_time = time.time()
    print(f"[워커 {worker_id}] 완료 ({end_time - start_time:.2f}초). 결과 {len(results)}개.")
    if results:
        print(f"[워커 {worker_id}] 첫 번째 결과: {results[0]['name']}")

async def main():
    """
    데모 실행과 자원 정리를 모두 처리하는 메인 함수.
    """
    try:
        search_terms = [
            "iPhone 14", "Samsung Galaxy S23", "MacBook Pro M3",
            "Sony WH-1000XM5", "LG OLED TV", "Nintendo Switch", "Dyson V15",
        ]

        # 서비스가 초기화되도록 보장합니다.
        await get_google_shopping_service()

        tasks = []
        for i, term in enumerate(search_terms):
            task = asyncio.create_task(worker(term, i + 1))
            tasks.append(task)
            # 실제 사용 환경과 유사하게 요청이 약간의 시차를 두고 들어오도록 설정
            await asyncio.sleep(0.2)

        await asyncio.gather(*tasks)

        print("\n모든 동시 검색 작업이 완료되었습니다.")
    finally:
        print("\n서비스를 종료합니다...")
        await close_google_shopping_service()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")