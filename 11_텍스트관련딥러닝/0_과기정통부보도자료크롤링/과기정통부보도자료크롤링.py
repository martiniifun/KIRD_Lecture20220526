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

# 브라우저(크롬드라이버) 열기
driver = webdriver.Chrome(driver_path, options=chrome_options)

# 과기정통부 보도자료 페이지 접속
driver.get("https://www.msit.go.kr/bbs/list.do?sCode=user&mPid=112&mId=113")

# (현재 1페이지이지만 확인차) 다시 1페이지로 이동
driver.execute_script("fn_paging(1);return false;")

# 첫 번째(최상단) 글 클릭
driver.find_element_by_css_selector('#td_NTT_SJ_0').click()

while True:
    try:  # 오류 핸들링을 위한 try-except 구문

        # 먼저 "다운로드"를 위한 <a> 태그 찾기
        download_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR,
                           """#search-view-form > div > div.view_file > ul > li > span > a:nth-child(1)""")))

        # <a href="javascript:void(0);" onclick="fn_download('38668', '1', 'hwpx');" class="down" title="파일다운로드">다운로드</a>
                                                #===================================

        # <a> 태그의 onclick 속성값 추출 : "fn_download('38668', '1', 'hwpx');"
        download_script = download_element.get_attribute('onclick')

        # onclick 속성(JS코드) 실행 -> 다운로드 시작
        driver.execute_script(download_script)

    except TimeoutException as e:  # 본문에 오류가 있는 경우(다운로드 없는 경우 간혹 있음) 프로그램이 종료되지 않게
        print(e)  # 오류메시지 출력
        print(driver.current_url)  # 첨부파일 없는 페이지 주소 출력

    try:
        # 다음글로 이동하는 onclick attribute(JS코드) 찾기
        pre_link_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#search-view-form > div > div.page_skip > ul > li:nth-child(2) > p > a')))
        pre_link_script = pre_link_element.get_attribute('onclick')

        # 다음글로 이동하는 JS코드 실행
        driver.execute_script(pre_link_script)

    except TimeoutException as e:  # 가끔 렉이 심하게 걸리거나, 페이지가 없는 경우 오류핸들링
        print(e)  # 오류메시지 출력하고
        driver.back()  # 다시 이전 페이지(이전글)로 이동
        sleep(3)  # 3초 후 재시도
        pre_link_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#search-view-form > div > div.page_skip > ul > li:nth-child(2) > p > a')))
        pre_link_script = pre_link_element.get_attribute('onclick')
        #%% # 이전글로 이동 코드 실행
        driver.execute_script(pre_link_script)

# 완료 코드는 따로 짜 두지 않았음
# "이전글" 버튼 없으면 알아서 꺼지겠지..