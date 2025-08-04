from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .crew import BlogsCrew
from .prompt_parser import PromptFormatter
from langchain_google_genai import ChatGoogleGenerativeAI 
import traceback
import asyncio
import json
import os
import time
import re

app = FastAPI(
    title="AI Social Blogging App Backend",
    description="An API for generating blog posts using a CrewAI multi-agent system."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  
        "http://localhost:5173", 
        "https://blogproject-production-b8ce.up.railway.app",
        "https://social-blogging-app-mauve.vercel.app"  
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class BlogGenerationRequest(BaseModel):
    topic: str
    tone: str = "professional"
    target_audience: str = "a general audience"

class FreestylePromptRequest(BaseModel):
    prompt: str

def get_prompt_formatter():
    return PromptFormatter(
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    )

def parse_crew_result(result, topic: str) -> dict:
    """
    Parse and normalize the crew result into a consistent format
    """
    try:
        if isinstance(result, dict):
            parsed_result = result
        elif isinstance(result, str):
            cleaned_result = result.strip()
            
            if cleaned_result.startswith('```json') and cleaned_result.endswith('```'):
                cleaned_result = cleaned_result[7:-3].strip()
            elif cleaned_result.startswith('```') and cleaned_result.endswith('```'):
                cleaned_result = cleaned_result[3:-3].strip()
            json_match = re.search(r'{.*}', cleaned_result, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group()
                    json_str = json_str.replace('\n', ' ').replace('\\n', '\n')
                    parsed_result = json.loads(json_str)
                    print(f"Successfully parsed JSON with keys: {list(parsed_result.keys())}")
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed: {e}")
                    parsed_result = {"full_content": cleaned_result}
            else:
                parsed_result = {"full_content": cleaned_result}
        else:
            parsed_result = {"full_content": str(result)}

        normalized = {
            "seo_title": None,
            "title": None,
            "meta_description": None,
            "summary": None,
            "hashtags": [],
            "full_content": None
        }

        field_mappings = {
            "seo_title": ["seo_title", "title", "headline"],
            "title": ["title", "seo_title", "headline"],
            "meta_description": ["meta_description", "description", "desc"],
            "summary": ["summary", "brief", "overview"],
            "hashtags": ["hashtags", "tags", "keywords"],
            "full_content": ["full_content", "content", "blog_post", "article", "body"]
        }

        for norm_key, possible_keys in field_mappings.items():
            for key in possible_keys:
                if key in parsed_result and parsed_result[key]:
                    normalized[norm_key] = parsed_result[key]
                    break

        if not any(normalized.values()):
            normalized["full_content"] = str(result)
            normalized["title"] = f"Blog Post: {topic}"

        if not normalized["title"] and not normalized["seo_title"]:
            normalized["title"] = f"Blog Post about {topic}"

        if normalized["hashtags"] and isinstance(normalized["hashtags"], str):
            hashtags_str = normalized["hashtags"]
            if ',' in hashtags_str:
                normalized["hashtags"] = [tag.strip() for tag in hashtags_str.split(',')]
            elif '#' in hashtags_str:
                normalized["hashtags"] = re.findall(r'#\w+', hashtags_str)
            else:
                normalized["hashtags"] = [hashtags_str]

        return normalized

    except Exception as e:
        print(f"Error parsing crew result: {e}")
        return {
            "seo_title": f"Blog Post: {topic}",
            "title": f"Blog Post: {topic}",
            "meta_description": f"An informative blog post about {topic}",
            "summary": "Generated blog content",
            "hashtags": [f"#{topic.replace(' ', '')}"],
            "full_content": str(result)
        }

@app.get("/")
def read_root():
    return {"message": "AI Social Blogging App Backend is running!"}

@app.post("/api/generate-blog-from-prompt", tags=["Blog Generation"])
async def generate_blog_from_prompt(
    request: FreestylePromptRequest,
    formatter: PromptFormatter = Depends(get_prompt_formatter)
):
    start_time = time.time()
    
    try:
        print(f"Received freestyle prompt: '{request.prompt}'")
        structured_input = formatter.format_prompt(request.prompt)
        print(f"Structured input: {structured_input}")

        crew_request = BlogGenerationRequest(
            topic=structured_input.topic,
            tone=structured_input.tone,
            target_audience=structured_input.target_audience
        )

        crew_result = await generate_blog_internal(crew_request)
        execution_time = time.time() - start_time

        parsed_result = parse_crew_result(crew_result, structured_input.topic)
        response = {
            "result": parsed_result,
            "execution_time": execution_time,
            "status": "success",
            "metadata": {
                "original_prompt": request.prompt,
                "topic": structured_input.topic,
                "tone": structured_input.tone,
                "target_audience": structured_input.target_audience
            }
        }

        print(f"Returning response with keys: {list(response.keys())}")
        print(f"Result keys: {list(parsed_result.keys())}")
        
        return response

    except ValueError as ve:
        execution_time = time.time() - start_time
        error_response = {
            "result": None,
            "execution_time": execution_time,
            "status": "error",
            "error": str(ve),
            "error_type": "validation_error"
        }
        raise HTTPException(status_code=400, detail=error_response)
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"An error occurred after {execution_time:.1f}s: {e}")
        traceback.print_exc()
        
        error_response = {
            "result": None,
            "execution_time": execution_time,
            "status": "error",
            "error": str(e),
            "error_type": "processing_error"
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.post("/api/generate-blog", tags=["Blog Generation"])
async def generate_blog(request: BlogGenerationRequest):
    """Public endpoint for direct blog generation"""
    start_time = time.time()
    
    try:
        crew_result = await generate_blog_internal(request)
        execution_time = time.time() - start_time
        
        parsed_result = parse_crew_result(crew_result, request.topic)
        
        return {
            "result": parsed_result,
            "execution_time": execution_time,
            "status": "success",
            "metadata": {
                "topic": request.topic,
                "tone": request.tone,
                "target_audience": request.target_audience
            }
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"An error occurred after {execution_time:.1f}s: {e}")
        traceback.print_exc()
        
        error_response = {
            "result": None,
            "execution_time": execution_time,
            "status": "error",
            "error": str(e),
            "error_type": "processing_error"
        }
        raise HTTPException(status_code=500, detail=error_response)

async def generate_blog_internal(request: BlogGenerationRequest):
    """Internal function for blog generation logic"""
    print(f"Generating blog for topic: '{request.topic}', tone: '{request.tone}', audience: '{request.target_audience}'")
    
    crew_setup = BlogsCrew(
        topic=request.topic,
        tone=request.tone,
        target_audience=request.target_audience
    )

    if not crew_setup.rag_tool and not crew_setup.serpapi_tool:
        raise HTTPException(
            status_code=500, 
            detail="No knowledge tools available (RAG and SerpAPI failed)."
        )

    blog_crew = crew_setup.setup_crew()
    print("Starting crew execution...")
    
    try:
        result = await asyncio.to_thread(blog_crew.kickoff)
        print("Crew execution finished successfully.")
        print(f"Raw result type: {type(result)}")
        print(f"Raw result preview: {str(result)[:200]}...")
        
        return result
        
    except Exception as e:
        print(f"Crew execution failed: {e}")
        raise e

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Social Blogging App Backend",
        "timestamp": time.time()
    }

