import requests
import json
from tkinter import *

root = Tk()
root.geometry("650x410")
root.title("네이버AI를 활용한 3줄요약 프로그램")


def take_input():
    INPUT = inputtxt.get("1.0", "end-1c")
    text = summ(INPUT)
    Output.insert(END, text.replace("\n", "\n\n"))


def summ(text, length=3, model="news", tone=3):
    # URL
    url = "https://naveropenapi.apigw.ntruss.com/text-summary/v1/summarize"

    # headers
    headers = {
        "X-NCP-APIGW-API-KEY-ID": "96d1m5ylby",
        "X-NCP-APIGW-API-KEY": "sKO97t3GewTMlLUVGAKHtB1hsiYpPvacJjmJO2hP",
        "Content-Type": "application/json"
    }

    # data
    data = {
      "document": {
        # "title": "개인맞춤형정보(마이데이터)에 새로운 가치를 더하다!",
        "content": text[:2000].strip()
      },
      "option": {
        "language": "ko",  # "ja": 일본어
        "model": model,  # "general": 일반 문서 요약, "news": 뉴스 요약
        "tone": tone,  # 0: 원문 어투를 유지, 1: 해요체로 변환, ex) 조사한다 -> 조사해요, 2: 정중체로 변환, ex)조사한다 -> 조사합니다 3: 명사형 종결체로 변환, ex) 조사한다 -> 조사함
        "summaryCount": length  # 요약된 문서의 문장수, 미지정시 기본 값은 3
      }
    }

    # data 딕셔너리를 JSON으로 변환
    data = json.dumps(data)


    response = requests.post(url, headers=headers, data=data)

    # print("response: ", response)
    return json.loads(response.text)["summary"]


if __name__ == '__main__':
    label = Label(text="기사 전문을 붙여넣기하세요.")
    inputtxt = Text(root, height=10,
                    width=75,
                    bg="light yellow")

    Output = Text(root, height=10,
                  width=75,
                  bg="light cyan")

    Display = Button(root, height=2,
                     width=20,
                     text="3줄 요약하기",
                     command=lambda: take_input())

    label.pack()
    inputtxt.pack()
    Display.pack()
    Output.pack()

    mainloop()
