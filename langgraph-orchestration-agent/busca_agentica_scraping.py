import os
import re
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from tavily import TavilyClient
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()

api_key = os.environ.get("TAVILY_API_KEY")

cidade = "Belém do Pará"
tavily_query = f"restaurantes em {cidade} tripadvisor com maior quantidade de reviews e faixa de preço"

print("Iniciando busca agêntica por URLs do Tripadvisor com Tavily...")

tripadvisor_url = None

if not api_key or api_key == "SUA_CHAVE_AQUI":
    print("Aviso: Chave TAVILY_API_KEY não configurada no .env.")
else:
    try:
        cliente_tavily = TavilyClient(api_key=api_key)
        tavily_results = cliente_tavily.search(query=tavily_query, max_results=5)

        if tavily_results and tavily_results.get("results"):
            print(f"Tavily encontrou {len(tavily_results['results'])} resultados. Analisando...")

            for result in tavily_results["results"]:
                url = result["url"]
                if "tripadvisor.com" in url or "tripadvisor.com.br" in url:
                    tripadvisor_url = url
                    break

            if not tripadvisor_url:
                print("Nenhum URL relevante do Tripadvisor foi encontrado nos primeiros resultados.")
        else:
            print("Tavily não encontrou resultados para a busca agêntica.")
    except Exception as e:
        print(f"Erro na busca agêntica com Tavily: {e}. Verifique sua chave API ou conexão.")

if tripadvisor_url:
    clean_url = re.sub(r'-o\d+-', '-', tripadvisor_url)
    tripadvisor_url = clean_url
    print("✅ URL encontrada limpa de paginação.")

print("-" * 50)
print(f"URL Final do Tripadvisor para raspagem: {tripadvisor_url if tripadvisor_url else 'NÃO ENCONTRADO'}")
print("-" * 50)

def scrape_restaurantes_info(url):
    if not url:
        print("Erro: URL vazia ou não localizada para raspagem.")
        return None
    try:
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126")

        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(30)

        print(f"Tentando carregar a página com Selenium: {url}")
        driver.get(url)
        driver.implicitly_wait(10)

        response_text = driver.page_source
        driver.quit()

        soup = BeautifulSoup(response_text, "html.parser")
        return soup
    except Exception as e:
        print(f"Erro ao raspar com Selenium: {e}")
        return None

soup_tripadvisor = None

if tripadvisor_url:
    print(f"\nTentando raspar a página identificada: {tripadvisor_url}")
    soup_tripadvisor = scrape_restaurantes_info(tripadvisor_url)
    if soup_tripadvisor:
        print("HTML da página do Tripadvisor obtido com sucesso!")
        page_title_tag = soup_tripadvisor.find("title")
        if page_title_tag:
            print(f"Título da página: {page_title_tag.get_text(strip=True)}")

if soup_tripadvisor:
    restaurant_blocks = soup_tripadvisor.find_all("div", class_="tbrcR")
    print(f"\nIniciando a extração detalhada de cada restaurante (encontrados: {len(restaurant_blocks)}):")
    restaurantes_detalhados = []

    for block in restaurant_blocks:
        nome_link_tag = block.find("a", class_=lambda c: c and "BMQDV" in c and "ukgos" in c)
        nome = "Nome não encontrado"
        if nome_link_tag:
            nome_div_tag = nome_link_tag.find("div", class_=lambda c: c and "biGQs" in c and "fiohW" in c)
            if nome_div_tag:
                nome = nome_div_tag.get_text(strip=True)

        reviews_tag = block.find("div", {"data-automation": "bubbleReviewCount"})
        reviews = reviews_tag.find("span").get_text(strip=True) if reviews_tag and reviews_tag.find("span") else "Reviews não encontrado"

        rating_tag = block.find("div", {"data-automation": "bubbleRatingValue"})
        rating = rating_tag.find("span").get_text(strip=True) if rating_tag and rating_tag.find("span") else "Avaliação não encontrada"

        culinaria_preco_div = block.find("div", class_=lambda c: c and "ZvrsW" in c and "biqBm" in c)
        tipo_culinaria = "Tipo de Culinária não encontrado"
        preco = "Preço não encontrado"

        if culinaria_preco_div:
            spans_info = culinaria_preco_div.find_all("span", class_=lambda c: c and "biGQs" in c and "pZUbB" in c)
            if len(spans_info) > 1:
                tipo_culinaria = spans_info[0].get_text(strip=True)
            for i in range(1, len(spans_info)):
                text = spans_info[i].get_text(strip=True)
                if "$" in text:
                    preco = text
                    break

        localizacao = "Localização não especificada (do nome)"
        if " - " in nome:
            partes_nome = nome.split(" - ")
            if len(partes_nome) > 1:
                localizacao = partes_nome[-1].strip()

        link = "Link não encontrado"
        if nome_link_tag and "href" in nome_link_tag.attrs:
            link = "https://www.tripadvisor.com.br" + nome_link_tag["href"]

        restaurantes_detalhados.append({
            "Nome": nome,
            "Avaliacao": rating,
            "Reviews": reviews,
            "Preco": preco,
            "Tipo Culinária": tipo_culinaria,
            "Localizacao": localizacao,
            "Link": link
        })

    if restaurantes_detalhados:
        print(f"\n--- {len(restaurantes_detalhados)} Restaurantes Extraidos do Tripadvisor ---")
        for i, r in enumerate(restaurantes_detalhados):
            print(f"--- Restaurante #{i+1} ---")
            for key, value in r.items():
                print(f"{key}: {r[key]}")
            print("-" * 40)
    else:
        print("Nenhum detalhe de restaurante foi extraido.")
else:
    print("\nNenhuma raspagem foi executada pois não houve URL capturada.")
