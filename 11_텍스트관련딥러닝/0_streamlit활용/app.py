from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st


with st.echo(code_location='below'):
    total_points = st.slider("스파이럴의 점 갯수", 1, 5000, 2000)
    num_turns = st.slider("스파이럴의 스핀 수", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))


# 1. 깃을 설치해서 커밋
# 2. 푸쉬(깃헙 계정이 있어야 함, 레포지토리 생성)
# 3. 스트림릿에 레포 주소 및 .py 파일 위치 입력
# 4. requirements.txt