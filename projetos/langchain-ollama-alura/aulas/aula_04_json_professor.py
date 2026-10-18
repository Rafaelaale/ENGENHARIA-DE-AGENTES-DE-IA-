from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from langchain_core.globals import set_debug
from dotenv import load_dotenv
import os

set_debug(True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class Destino(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar")
    motivo: str = Field("motivo pelo qual é interessante visitar essa cidade")

parseador = JsonOutputParser(pydantic_object=Destino)

prompt_cidade = PromptTemplate(
    template="""
Sugira uma cidade dado o meu interesse por {interesse}.
{formato_de_saida}
""",
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador.get_format_instructions()}
)

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

cadeia = prompt_cidade | modelo | parseador

resposta = cadeia.invoke({"interesse": "praias"})
print(resposta)

