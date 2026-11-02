import math
import os
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:3b"
KNOWLEDGE_FILE = "knowledge.txt"


def get_embedding(text):
  """Ollama의 임베딩 API를 이용해 텍스트를 벡터로 변환"""
  try:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": MODEL_NAME, "prompt": text},
    )
    return response.json().get("embedding", [])
  except Exception as e:
    print(f"⚠️ 임베딩 생성 오류: {e}")
    return []


def cosine_similarity(v1, v2):
  """두 벡터 사이의 코사인 유사도 계산"""
  if not v1 or not v2 or len(v1) != len(v2):
    return 0.0
  dot_product = sum(a * b for a, b in zip(v1, v2))
  magnitude1 = math.sqrt(sum(a * a for a in v1))
  magnitude2 = math.sqrt(sum(a * a for a in v2))
  if magnitude1 == 0 or magnitude2 == 0:
    return 0.0
  return dot_product / (magnitude1 * magnitude2)


class StrictFileRAGChat:

  def __init__(self):
    self.memory = []
    print("🧠 순수 파이썬 RAG 엔진 초기화 중...")
    self.load_knowledge_from_file()

  def load_knowledge_from_file(self):
    """외부 텍스트 파일에서 지식을 읽어와 벡터화"""
    if not os.path.exists(KNOWLEDGE_FILE):
      print(f"⚠️ '{KNOWLEDGE_FILE}' 파일이 없습니다. 빈 파일로 생성합니다.")
      with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        f.write(
            "민지욱은 대치동 및 분당 지역의 수학/과학 학원을 알아보며 인공지능과"
            " 하드웨어를 공부한다.\n"
        )

    print(f"📚 '{KNOWLEDGE_FILE}'에서 지식을 읽어오는 중...")
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
      # 문장 단위나 줄 단위로 분리 (빈 줄 제외)
      lines = [
          line.strip()
          for line in f.readlines()
          if line.strip() and not line.startswith("#")
      ]

    for idx, text in enumerate(lines):
      emb = get_embedding(text)
      if emb:
        self.memory.append({"id": f"doc_{idx}", "text": text, "embedding": emb})

    print(
        f"✨ 총 {len(self.memory)}개의 지식 조각이 벡터 메모리에 적재되었습니다!\n"
    )

  def search_knowledge(self, query):
    """질문과 가장 유사한 지식 문서 검색 (상위 2개)"""
    if not self.memory:
      return []

    query_emb = get_embedding(query)
    if not query_emb:
      return []

    scored_docs = []
    for item in self.memory:
      score = cosine_similarity(query_emb, item["embedding"])
      scored_docs.append((score, item["text"]))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc_text for score, doc_text in scored_docs[:2]]

  def chat_loop(self):
    print(
        f"🔥 [엄격한 검색 기반 RAG 활성화됨] {MODEL_NAME}와의 대화가"
        " 시작되었습니다."
    )
    print("종료하려면 'exit' 또는 'quit'을 입력하세요.\n" + "=" * 50)

    conversation_history = []

    while True:
      try:
        user_input = input("\n👤 사용자: ")
      except (KeyboardInterrupt, EOFError):
        print("\n대화를 종료합니다.")
        break

      if user_input.lower() in ["exit", "quit"]:
        print("대화를 종료합니다.")
        break

      if not user_input.strip():
        continue

      # 1. 외부 파일 기반 RAG 검색
      retrieved_docs = self.search_knowledge(user_input)

      context_text = ""
      if retrieved_docs:
        context_text = "\n".join([f"- {doc}" for doc in retrieved_docs])
        print(f"\n[🔍 RAG 검색된 외부 지식]\n{context_text}")

      conversation_history.append(f"User: {user_input}")
      full_context = "\n".join(conversation_history[-6:])

      # 2. 모델 자체 지식 차단 및 검색된 문서만 강제하는 프롬프트
      prompt = f"""
      당신은 자체적인 백과사전적 지식이 전혀 없는 순수 언어 처리기입니다. 
      당신에게는 오직 아래의 [참고 지식]만이 유일한 진실입니다. 

      규칙:
      1. 답변은 반드시 아래의 [참고 지식] 내용만을 기반으로 작성하세요.
      2. [참고 지식]에 없는 내용은 "가지고 있는 지식 파일에 해당 내용이 없습니다"라고 솔직하게 답하세요.
      3. 당신의 기존 사전 지식이나 추측을 절대로 섞어서 답변하지 마세요.

      [참고 지식 (RAG)]:
      {context_text if context_text else "관련된 참고 지식 없음"}

      [최근 대화 기록]:
      {full_context}
      
      AI:
      """

      try:
        print("🤖 AI 생각 중...", end="\r")
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=60,
        )
        reply = response.json().get("response", "").strip()

        print(f"🤖 AI: {reply}")
        conversation_history.append(f"AI: {reply}")

      except Exception as e:
        print(f"\n⚠️ 오류 발생: {e}")


if __name__ == "__main__":
  agent = StrictFileRAGChat()
  agent.chat_loop()