# Tavily
from langchain_tavily import TavilySearch, TavilyExtract

tavily_extract = TavilyExtract()
tavily_search = tool = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=True,
    include_raw_content=True,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)


url = "https://learngerman.dw.com/de/uhrzeit-informell-2/l-40443235/gr-40445046"


# Funktion, die den Text von der Seite extrahiert
def extract_website_content(url):
    extract_results = tavily_extract.invoke({"urls": [url]})
    for result in extract_results["results"]:
        extract = result["raw_content"]
    # print("FERTIG")
    return extract_results


def get_website_informations(url):
    # search_results = tavily_search.invoke(url)
    # search_results = tavily_search.invoke(url,{"query": "What are the Infromal Time rules in german language?"})
    search_results = tavily_search.invoke(
        url,
        {"query": "What are the Infromal Time rules in german language on this page?"},
    )
    # search_results = tavily_search.invoke({"query": "What are the Infromal Time rules in german language?"})
    for result in search_results["results"]:
        print(result["content"])
        content = result["content"]
    return search_results
def get_website_ctx(url):
    # search_results = tavily_search.invoke(url)
    # search_results = tavily_search.invoke(url,{"query": "What are the Infromal Time rules in german language?"})
    search_results = tavily_search.invoke(
        url,
    )
    # search_results = tavily_search.invoke({"query": "What are the Infromal Time rules in german language?"})
    for result in search_results["results"]:
        print(result["content"])
        content = result["content"]
    return search_results


# output = get_website_informations(url)
# print(output["results"])
# print("NEXT OUTPUT")
# print(output["results"][0])
# print(output["results"][0].get("content"))
# output = get_website_ctx(url)
# print(output["results"])
# print("NEXT OUTPUT")
# print(output["results"][0])
# print(output["results"][0].get("content"))

# result = extract_website_content(url)
# prompt_time_knowledge = result.get("results")[0].get("raw_content")
# prompt_time_knowledge = result.get("results")[0].get("raw_content")

# print(prompt_time_knowledge)
