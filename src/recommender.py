from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0.4
        )

        # Your custom prompt (must be ChatPromptTemplate now)
        self.prompt = get_anime_prompt()

        # Build modern RAG chain
        self.qa_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | RunnablePassthrough.assign(
        answer=self.prompt | self.llm | StrOutputParser()
    )
)

    def get_recommendation(self, query: str):
        return self.qa_chain.invoke(query)