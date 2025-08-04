from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyCustomToolInput(BaseModel):
    query: str = Field(..., description="The search query for the knowledge base.")

class MyCustomTool(BaseTool):
    name: str = "RAG Search Tool"
    description: str = "Retrieves context from blog post examples and style guides to help with content creation."
    args_schema: Type[BaseModel] = MyCustomToolInput
    api_key: str 
    embedding_model: GoogleGenerativeAIEmbeddings = None
    vector_db = None
    is_initialized: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
            self._initialize_db()
            logger.info("RAG Search Tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Search Tool: {e}")
            self.is_initialized = False

    def _initialize_db(self):
        """Initialize the vector database with multiple possible paths"""
        possible_paths = [
            "faiss_index",
            "./faiss_index", 
            "/app/faiss_index",
            os.path.join(os.getcwd(), "faiss_index"),
            os.path.join(os.path.dirname(__file__), "..", "faiss_index"),
        ]
        
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Directory contents: {os.listdir('.')}")
        
        for path in possible_paths:
            logger.info(f"Checking path: {path}")
            if os.path.exists(path):
                try:
                    logger.info(f"Found FAISS index at: {path}")
                    self.vector_db = FAISS.load_local(
                        path,
                        embeddings=self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                    self.is_initialized = True
                    logger.info(f"Successfully loaded FAISS index from {path}")
                    return
                except Exception as e:
                    logger.error(f"Failed to load FAISS index from {path}: {e}")
                    continue
        
        logger.warning("No FAISS index found in any of the expected locations")
        self.is_initialized = False

    def _run(self, query: str) -> str:
        """
        Execute the RAG search with the provided query.
        
        Args:
            query (str): The search query string
            
        Returns:
            str: Retrieved context or error message
        """
        logger.info(f"RAG Search Tool called with query: '{query}'")
        
        if not self.is_initialized or self.vector_db is None:
            error_msg = "RAG Search Tool is not properly initialized. Knowledge base not available."
            logger.error(error_msg)
            return error_msg
        
        try:
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            
            if not docs:
                logger.warning(f"No documents found for query: '{query}'")
                return f"No relevant information found in the knowledge base for query: '{query}'"
            
            context = "\n---\n".join([doc.page_content for doc in docs])
            logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
            return f"Retrieved context for '{query}':\n{context}"
            
        except Exception as e:
            error_msg = f"Error retrieving from knowledge base: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _arun(self, query: str) -> str:
        """Async version of the run method."""
        return self._run(query)

    def is_available(self) -> bool:
        """Check if the tool is available for use"""
        return self.is_initialized and self.vector_db is not None
        
