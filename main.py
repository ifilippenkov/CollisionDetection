# python3 /path/to/text-preprocessing/main.py --input_json /path/to/text-preprocessing/input.json

from src.graph_rag import custom_embedder, custom_llm
import json
from src import chunk_getter
import argparse
from llama_index.core import PropertyGraphIndex
from src.fact_checker import checker
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.graph_stores.neo4j import Neo4jPGStore
import tokens

from llama_index.core.retrievers import VectorContextRetriever, LLMSynonymRetriever
from llama_index.core import QueryBundle

def get_retrieved_nodes(
    index, custom_llm, custom_embedder, query_str, vector_top_k=10, reranker_top_n=3, with_reranker=False
):
    query_bundle = QueryBundle(query_str)

    syn = LLMSynonymRetriever(
        graph_store=index.property_graph_store,
        llm=custom_llm,
        include_text=True,
        max_keywords=8,
        path_depth=5
    )

    vec = VectorContextRetriever(
        graph_store=index.property_graph_store,
        vector_store=index.vector_store,
        embed_model=custom_embedder,
        include_text=True,
        similarity_top_k=vector_top_k,
        path_depth=5
    )

    retriever = index.as_retriever(sub_retrievers=[syn, vec], include_text=True)

    retrieved_nodes = retriever.retrieve(query_bundle)
    if with_reranker:
        prompt_str = (
            "A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided. \n"
            "Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well as the relevance score. The relevance score is a number from 1-10 based on how relevant you think the document is to the question.\n"
            "Prioritize documents based on their relevance to the question, regardless of whether they support or contradict the query. Both confirming and contradicting facts are considered equally relevant if they provide significant information, context, or arguments related to the question.\n"
            "Assign relevance scores in a balanced way to fairly represent differing viewpoints or data, ensuring that conflicting evidence is not overshadowed by other documents.\n"
            "Always include at least one document in the response, selecting the most relevant documents even if the relevance is low.\n"
            "Do not include documents that are irrelevant to the question.\n"
            "Example format: \n"
            "Document 1:\n<summary of document 1>\n\n"
            "Document 2:\n<summary of document 2>\n\n"
            "...\n\n"
            "Document 10:\n<summary of document 10>\n\n"
            "Question: <question>\n"
            "Answer:\n"
            "Doc: 9, Relevance: 7\n"
            "Doc: 3, Relevance: 4\n"
            "Doc: 7, Relevance: 3\n\n"
            "Let's try this now: \n\n"
            "{context_str}\n"
            "Question: {query_str}\n"
            "Answer:\n"
        )

        custom_choice_template = PromptTemplate(
            template=prompt_str
        )

        reranker = LLMRerank(
            llm=custom_llm,
            choice_batch_size=5,
            top_n=reranker_top_n,
            choice_select_prompt=custom_choice_template
        )

        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
    return retrieved_nodes


def main():
    parser = argparse.ArgumentParser(description="Collision detection.")
    parser.add_argument("--input_json", type=str, help="Path to input json")
    parser.add_argument("--has_graph", type=bool, default=False, help="Exist graph")
    parser.add_argument("--language", type=str, default="en", choices=["en", "ru"], 
                        help="Language for system prompt: en (English) or ru (Russian)")

    args = parser.parse_args()

    with open(args.input_json, "r") as file:
        input_json = json.load(file)
        data = chunk_getter.Data(input_json["conflict"], input_json["data_path"], chunker=input_json["chunker"])

    nodes = data.node_getter()

    graph_store = Neo4jPGStore(url=tokens.NEO4J_URL, username=tokens.NEO4J_USERNAME, password=tokens.NEO4J_PASSWORD)
    graph_index = None
    if args.has_graph:
        graph_index = PropertyGraphIndex.from_existing(
            llm=custom_llm,
            property_graph_store=graph_store,
            embed_model=custom_embedder,
            include_embeddings=True,
        )
    else:
        graph_index = PropertyGraphIndex(
            nodes=nodes,
            llm=custom_llm,
            property_graph_store=graph_store,
            embed_model=custom_embedder,
            include_embeddings=True,
        )

    retrieved_nodes = get_retrieved_nodes(graph_index, custom_llm, custom_embedder, data.input_promt, vector_top_k=30, reranker_top_n=5, with_reranker=True)

    facts = []
    for node in retrieved_nodes:
        text = str(node.node.get_text())
        facts.append(text)

    # Создаем checker с выбранным языком
    from src.llm_service import YandexCloudLLM
    from src.fact_checker import FactConsistencyChecker
    llm_service = YandexCloudLLM(
        api_key=tokens.AUTH_TOKEN,
        folder_id=tokens.FOLDER_ID,
        model_uri="yandexgpt-lite",
        temperature=0.1
    )
    language_checker = FactConsistencyChecker(llm_service=llm_service, language=args.language)
    
    result = language_checker.check_facts(data.input_promt, facts)
    print(result)

if __name__ == '__main__':
    import time

    start_time = time.time()
    main()
    end_time = time.time()

    execution_time_seconds = end_time - start_time
    execution_time_minutes = execution_time_seconds / 60

    print(f"Функция выполнилась за {execution_time_minutes:.2f} минут")
