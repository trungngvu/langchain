from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field

class Search(BaseModel):
    """Search over a database of computers' parts"""
    gpu: Optional[str] = Field(None, description="Name of your recommended Graphics card")
    cpu: Optional[str] = Field(None, description="Name of your recommended Processor")
    CPU_socket: Optional[str] = Field(None, description="The CPU Socket of the CPU you recommended")
    RAM: Optional[str] = Field(None, description="Your recommended amount of RAM in Gigabytes")
    RAM_type: Optional[str] = Field(None, description="Your recommended type of RAM in one of the following: DDR3, DDR4, DDR5")
    PSU: Optional[str] = Field(None, description="Your recommended amount of the Power supply wattage")
    SSD_size: Optional[str] = Field(None, description="Your recommended amount of the Solid State Drive capacity, output in either GB or TB")
    HDD_size: Optional[str] = Field(None, description="Your recommended amount of the Hard Disk Drive capacity, output in either GB or TB")
    Case: Optional[str] = Field(None, description="Your recommended size of the PC case size in one of the following: [small, medium, standard, large]")
    Cooler: Optional[str] = Field(None, description="Your recommended type of CPU cooler in one of the following: [air, water]")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import json

system = """You are an expert at converting user questions into database queries and an expert PC builder. \
You have access to a database of computers' parts including GPU, CPU, hard disks, PSU, Motherboard, RAM. \
If the user wants you to recommend something, try to use real and recent computer's part in your response. \
You should consider between Intel, AMD, and Nvidia CPUs and GPUs. \
Try spending the least amount of money as posible if the application does not require much. \
If the user have specify requirement for a specific application, try to use Recommended System Requirements for that app. \
All your response should be compatible with all the others. \
If the user given a budget in Vietnam Dong, you can convert it to US Dollar by dividing it with 24000. \
Given a question, return a list of database queries optimized to retrieve the most relevant results."""
llm = ChatGoogleGenerativeAI(temperature=0, google_api_key='AIzaSyAYtOXU6B6ubwRjk0-f8Lq_yvM-abbdm5U', model="gemini-1.5-flash")
# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Search)

prompt = PromptTemplate(
    template= system + "\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
 

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

MONGODB_ATLAS_CLUSTER_URI = 'mongodb+srv://nghia:nghia2002@cluster0.4miytl4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
DB_NAME = "tronicsify"
COLLECTION_NAME = "products"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
from bson import ObjectId  # Import ObjectId from the appropriate package

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_ATLAS_CLUSTER_URI,
    DB_NAME + "." + COLLECTION_NAME,
    HuggingFaceInferenceAPIEmbeddings(api_key = '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', model_name="sentence-transformers/all-MiniLM-L12-v2"),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key = "title",
    embedding_key = "embedding"
)

from typing import List
from langchain_core.documents import Document

def size(s):
    if s=="small": return "ITX"
    if s=="medium": return "M-ATX"
    if s=="large": return "E-ATX"
    return "ATX"

def cooler(s):
    if s=="water" or s=="liquid": return "aio"
    return "air"

def socket(s):
    if "LGA" in s:
      s = s.replace("LGA", "")
    s.strip()
    return s

def convert_objid(docs):
    res = []
    for doc in docs:
        new_doc = {}
        new_doc['page_content'] = doc.page_content
        new_doc['metadata'] = doc.metadata
        new_doc['metadata']["_id"] = str(new_doc['metadata']["_id"])
        if new_doc['metadata'].get("gpu"): del new_doc['metadata']["gpu"]
        if new_doc['metadata'].get('cpu'): del new_doc['metadata']["cpu"] 
        res.append(new_doc)
    return res

def retriever(search: Search) -> List[Document]:
    docs =[]
    if search.gpu: docs.append(convert_objid(vector_search.similarity_search(search.gpu, k=5, pre_filter = {"category": "gpu", "buildable": True})))
    if search.cpu: docs.append(convert_objid(vector_search.similarity_search(search.cpu, k=5, pre_filter = {"category": "cpu", "buildable": True})))
    if search.RAM: docs.append(convert_objid(vector_search.similarity_search("RAM " + search.RAM + "GB", k=5, pre_filter = {"category": "ram", "ram": search.RAM_type, "buildable": True})))
    if search.SSD_size: docs.append(convert_objid(vector_search.similarity_search("SSD " + search.SSD_size, k=5, pre_filter = {"category": "disk", "sub_category": "ssd", "buildable": True})))
    if search.HDD_size: docs.append(convert_objid(vector_search.similarity_search("HDD " + search.HDD_size, k=5, pre_filter = {"category": "disk", "sub_category": "hdd", "buildable": True})))
    if search.PSU: docs.append(convert_objid(vector_search.similarity_search("Nguồn " + search.PSU + "W", k=5, pre_filter = {"category": "psu", "buildable": True})))
    if search.Case:
      docs.append(convert_objid(vector_search.similarity_search("Mainboard " + size(search.Case), k=5, pre_filter = {"category": "main", "ram": search.RAM_type, "buildable": True, "socket": socket(search.CPU_socket), "size": size(search.Case)})))
      docs.append(convert_objid(vector_search.similarity_search("Vỏ case máy tính " + size(search.Case), k=5, pre_filter = {"category": "case", "buildable": True, "size": size(search.Case)})))
    if search.Cooler: docs.append(convert_objid(vector_search.similarity_search("Tản nhiệt " + cooler(search.Cooler) , k=5, pre_filter = {"category": "cooler", "sub_category": cooler(search.Cooler)})))
    return docs

def format_docs(docss):
  res = ""
  for docs in docss:
    for doc in docs:
      res += f"id:{doc.get('metadata').get('_id')} {doc.get('page_content')} {str(doc.get('metadata').get('price'))}VND\n"
  return res
template = """You are an expert PC builder. \
You will be given computers' parts including Graphics Cards, Processors, hard disks, PSUs, Motherboards, RAMs, CPU coolers. \
If the user have specify requirement for a specific application, try to use Recommended System Requirements for that app. \
All your components you recommend should be compatible with all the others. \
You should consider whether or not to include a Hard Disk Drive (HDD) along with a SSD. \
If user's question is specific, say user want a specific type of graphics card, just recommend the graphics card but not the whole build. \

List of components to recommend in the following format 'id name price":
{context}

Question: {question}
Explicitly tell the reason why you choose those parts at the end of your response. \
Response entirely in Vietnamese. \
"""
final_prompt = ChatPromptTemplate.from_template(template)

# user_query = "Build cấu hình sản xuất video Youtube"

def parseJson(s):
    if s.content.find("\`\`\`\n\`\`\`") != -1:
        s.content = s.content.replace("\`\`\`\n\`\`\`", "\`\`\`")
    return s

def similarity_retriever(input): 
    input_str = input.messages[0].content.replace("'", '"')
    # Parse the JSON string
    data = json.loads(input_str)

    # Extract the value associated with the key 'query'
    result = data['query']
    docs = convert_objid(vector_search.similarity_search(result, k=10, pre_filter = {"buildable": True}))
    res = ""
    for doc in docs:
      res += f"{doc.get('metadata').get('_id')} "
    return res
     


similarity_template ="""{question}"""
similarity_prompt = ChatPromptTemplate.from_template(similarity_template)

add_routes(
    app,
    {"question": RunnablePassthrough(), "context": prompt | llm | parseJson | parser | retriever | format_docs}  | final_prompt | llm,
    path="/tronicsify",
)

add_routes(
   app,
    {"question": RunnablePassthrough(), "context": prompt} | similarity_prompt | similarity_retriever,
   path="/similar-products"
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
