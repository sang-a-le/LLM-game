import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==============================
# 한영 변환 매핑 (필요시 확장 가능)
# ==============================
KOR_ENG_MAP = {
    "트와이스": "twice",
    "방탄소년단": "bts",
    "소녀시대": "girls generation",
    "빅뱅": "bigbang",
    "치얼업": "cheer up",
    "다이너마이트": "dynamite",
    "러브스토리": "love story",
    "블랙핑크": "blackpink",
    "아이유": "iu",
    "엑소": "exo",
}


def normalize_text(text: str) -> str:
    """입력 문자열을 소문자, 특수문자 제거, 한글→영문 매핑"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9가-힣\s]", "", text)  # 영문, 한글, 숫자만 남김
    text = text.strip()

    # 한글 키워드 → 영어로 변환
    for kor, eng in KOR_ENG_MAP.items():
        if kor in text:
            text = text.replace(kor, eng)

    return text


# ==============================
# 1. LLM으로 퀴즈 문제 가져오기
# ==============================
def fetch_quiz(year: int, num_questions: int = 5):
    base_parser = JsonOutputParser()
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser, llm=ChatOpenAI(model="gpt-4o-mini")
    )

    prompt = ChatPromptTemplate.from_template("""
    너는 예능 '놀라운 토요일'의 피디야. 노래가사 퀴즈를 진행해.
    {year}년에 발표된 유명한 한국 노래 중 {num_questions} 곡을 무작위로 선택해서 반환해줘.
    각 항목은 'artist', 'title', 'lyric' (가사 두 줄) 세 필드를 포함해야해.
    가사 출력 형식은 반드시 JSON 형식으로 반환해줘. 
    무작위로 선택된 노래에서 가장 하이라이트인 노래 가사 두 줄 먼저 반환한 다음, 사용자의 응답에 따라 정답이나 오답을 반환해야해. 
    가사는 반드시 그 노래의 가사 내용이어야하고, 실제 노래 가사가 아닌 글을 반환해서는 안돼. 

    ### 출력 형식 ###
    !! JSON 형식으로만 반환하기
    - [노래 가사 두 줄] "lyric"
    - [사용자 답변] 
    - 정답일 경우, 정답! / 오답일 경우, 오답! 반환 + 'artist' - 'title'

    ### 예시 ###
    - 🤖 : 어떤가요 그대 당신도 나와 같나요 
    - 답변 : 넬 - 기억을 걷는 시간
    - 정답! 넬 - 기억을 걷는 시간

    단, 가사는 반드시 실제 그 노래의 가사 두 줄이어야 하고 너무 길지 않게 반환해.
    """)

    chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.8, api_key=OPENAI_API_KEY) | fixing_parser

    try:
        return chain.invoke({"year": year, "num_questions": num_questions})
        
        if isinstance(result, dict):
            result = list(result.values())

        return result
    
    except Exception as e:
        st.error(f"⚠️ LLM에서 문제를 불러오는 데 실패했습니다: {e}")
        return []


# ==============================
# 2. Streamlit 퀴즈 진행
# ==============================
def run_quiz_streamlit():
    quiz = st.session_state.quiz

    # 모든 문제를 다 풀면 결과 화면
    if st.session_state.finished:
        st.success(f"🎯 최종 점수: {st.session_state.score} / {len(quiz)*20}")
        st.markdown("### 📖 출제 문제 해설")
        for idx, q in enumerate(quiz, start=1):
            user_ans = st.session_state.answers[idx-1]
            st.write(f"{idx}. {q['artist']} - {q['title']} | 가사: {q['lyric']}")
            st.caption(f"내 답변: {user_ans}")

        # 🔄 게임 재시작 버튼
        if st.button("🔄 다시 시작"):
            for key in ["quiz", "current_q", "score", "finished", "feedback", "answers"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return

    # 현재 문제
    q_idx = st.session_state.current_q

    if q_idx < len(quiz):
        question = quiz[q_idx]
    else:
        st.error("⚠️ 문제를 불러오는 데 오류가 발생했습니다.")
        return

    st.subheader(f"문제 {q_idx+1}")
    st.write(f"가사: {question['lyric']}")

    # 사용자 입력
    user_answer = st.text_input("👉 가수 - 제목을 입력하세요", key=f"answer_{q_idx}")

    if st.button("제출", key=f"submit_{q_idx}"):
        correct = f"{question['artist']} - {question['title']}"
        st.session_state.answers.append(user_answer)

        # ✅ 정규화 비교 (영/한/대소문자/특수문자 무시)
        user_norm = normalize_text(user_answer)
        correct_norm = normalize_text(correct)

        if user_norm == correct_norm:
            st.session_state.score += 20
            st.session_state.feedback = f"✅ 정답! ({question['artist']} - {question['title']})"
        else:
            st.session_state.feedback = f"❌ 오답! 정답은 {question['artist']} - {question['title']}"

        # 다음 문제로 이동
        st.session_state.current_q += 1
        if st.session_state.current_q >= len(quiz):
            st.session_state.finished = True

        st.rerun()

    if st.session_state.feedback:
        st.info(st.session_state.feedback)


# ==============================
# 3. 메인 실행
# ==============================
def main():
    load_dotenv()

    st.title("🎶 연도별 노래 가사 퀴즈")

    if "quiz" not in st.session_state:
        year = st.number_input("퀴즈를 풀고 싶은 연도를 입력하세요:", min_value=1970, max_value=2025, step=1)

        if st.button("퀴즈 시작"):
            quiz = fetch_quiz(year, num_questions=5)
            if quiz:
                st.session_state.quiz = quiz
                st.session_state.current_q = 0
                st.session_state.score = 0
                st.session_state.finished = False
                st.session_state.feedback = ""
                st.session_state.answers = []
                st.rerun()
            else:
                st.warning("해당 연도의 문제를 가져올 수 없습니다.")
    else:
        run_quiz_streamlit()


if __name__ == "__main__":
    main()


# streamlit run C:\skn_17\LLM\05_rag\game_3.py