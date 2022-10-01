# !pip install selenium chromedriver_autoinstaller

from time import sleep

from selenium import webdriver
import chromedriver_autoinstaller
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from selenium.webdriver.chrome.options import Options


chrome_options = Options()
prefs = {'profile.default_content_setting_values.automatic_downloads': 1}
chrome_options.add_experimental_option("prefs", prefs)
driver_path = chromedriver_autoinstaller.install()
driver = webdriver.Chrome(options=chrome_options)

# 과기정통부 보도자료 페이지 접속
driver.get("https://www.msit.go.kr/bbs/list.do?sCode=user&mPid=112&mId=113")


# (확인차) 1페이지로 이동
driver.execute_script("fn_paging(1);")

# 첫 번째 글 클릭
driver.find_element(by=By.CSS_SELECTOR, value='#td_NTT_SJ_0').click()

while True:
    try:
        #%% # 다운로드 JS코드 찾기
        download_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-view-form > div > div.view_file > ul > li > span > a:nth-child(1)")))
        download_script = download_element.get_attribute('onclick')
        #%% # 다운로드 JS코드 실행
        driver.execute_script(download_script)
    except TimeoutException as e:  # 본문에 오류가 있는 경우(비어있음)
        print(e)  # 패스
        print(driver.current_url)

    try:
        pre_link_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#search-view-form > div > div.page_skip > ul > li:nth-child(2) > p > a')))
        pre_link_script = pre_link_element.get_attribute('onclick')
        driver.execute_script(pre_link_script)
    except TimeoutException as e:
        print(e)
        driver.back()
        sleep(3)
        pre_link_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#search-view-form > div > div.page_skip > ul > li:nth-child(2) > p > a')))
        pre_link_script = pre_link_element.get_attribute('onclick')
        driver.execute_script(pre_link_script)

