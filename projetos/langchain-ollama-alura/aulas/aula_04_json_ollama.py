from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from langchain_core.globals import set_debug

set_debug(True)

class Destino(BaseModel):
    cidade: str = Field(description="A cidade recomendada para visitar")
    motivo: str = Field(description="motivo pelo qual é interessante visitar essa cidade")

parseador = JsonOutputParser(pydantic_object=Destino)

prompt_cidade = PromptTemplate(
    template="""
Sugira uma cidade dado o meu interesse por {interesse}.
{formato_de_saida}
""",
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador.get_format_instructions()}
)

modelo = ChatOllama(model="llama3.2:1b", temperature=0.5)
cadeia = prompt_cidade | modelo | parseador
print(cadeia.invoke({"interesse": "praias"}))
