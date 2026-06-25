from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- (1) 文書の読み込み ---
loader = TextLoader("C:/Users/tshimazaki/Downloads/knowledge.txt", encoding="utf-8")
documents = loader.load()

# --- (2) チャンク分割 ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
)
chunks = splitter.split_documents(documents)
print(f"分割数: {len(chunks)}")

# --- (3) ベクトル化してDBに格納 ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings)

# --- (4) 検索器（Retriever）を作る ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# --- (5) LLMとプロンプトの準備 ---
llm = OllamaLLM(
    model="tinyllama",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
あなたは厳密な情報抽出システムです。

【絶対ルール】
- 必ず参考資料の文章からそのまま抜き出して答えること
- 書き換え・要約・翻訳は禁止
- 推測や一般知識の使用は禁止
- 資料に無い情報は絶対に作らない
- 無い場合は「資料には記載がありません」とだけ答える
- 回答は必ず日本語で出力すること

違反は禁止。

## 参考資料
{context}

## 質問
{question}
""")

# --- (6) 質問して回答する ---
def ask(question: str) -> str:
    docs = retriever.invoke(question)

    if not docs:
        return "資料には記載がありません"

    # そのまま返す（LLMを信用しない）
    return "\n\n".join(d.page_content for d in docs)


if __name__ == "__main__":
    q = "白百合女子大学について教えて？"
    print("質問:", q)
    print("回答:", ask(q))