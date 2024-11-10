from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class RAGSystem:
    def __init__(self):
        if not os.getenv("NVIDIA_API_KEY"):
            raise ValueError("NVIDIA_API_KEY environment variable is not set")
            
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            default_headers={"Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"}  # NVIDIA might require this
        )
        
        # Initialize embeddings and load vectorstore
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = FAISS.load_local(
            os.getenv("VECTORSTORE_PATH", "./vectorstore"),
            self.embeddings
        )

    def get_context(self, query: str, k: int = 3) -> str:
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in docs)

    async def generate_response(self, query: str) -> str:
        context = self.get_context(query)
        messages = [
            {
                "role": "system",
                "content": f"""Answer based on this context. If the answer isn't in the context, say so.
                Context: {context}"""
            },
            {"role": "user", "content": query}
        ]

        try:
            response = self.client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")