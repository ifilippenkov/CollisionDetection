from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import (
    SimpleNodeParser,
    SentenceSplitter
)
from yandex_cloud_ml_sdk import YCloudML
import  tokens

class Data:
    def __init__(self,
                 input_promt: str,
                 data_path: str,
                 format:str = "text",
                 chunker:str = "basic",
                 chunk_size = 500,
                 chunk_overlap = 75):
        self.input_promt = input_promt
        self.data_path = data_path
        self.format = format
        self.chunker = chunker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sdk = YCloudML(
          folder_id=tokens.FOLDER_ID,
          auth=tokens.AUTH_TOKEN,
        )

    def basic_chunks(self, docs: list[str]):
        base_splitter = SimpleNodeParser.from_defaults(chunk_size = self.chunk_size,
                                                       chunk_overlap = self.chunk_overlap)
        nodes = base_splitter.get_nodes_from_documents(docs)
        return nodes

    def llm_chunks(self, docs: list[str]):
        new_docs = []

        for doc in docs:
            messages = [
                {
                    "role": "user",
                    "text": "Rewrite the text so that each sentence contains exactly one fact. Leave the sequence of events unchanged",
                },
                {
                    "role": "user",
                    "text": str(doc),
                }
            ]
            new_chunks = (
                self.sdk.models.completions("yandexgpt-lite").configure(temperature=0.5).run(messages)
            )
            
            assert len(new_chunks.text) != 0, "no result"

            new_doc = new_chunks.text
            sents = new_doc.split(".")
            new_docs.extend([sent for sent in sents if sent.strip()])

        docs = [Document(text=doc) for doc in new_docs]
        parser = SentenceSplitter()
        return parser.get_nodes_from_documents(docs)

    def node_getter(self):
        docs = []
        if self.format == "text":
            docs = SimpleDirectoryReader(input_dir=self.data_path, recursive=True).load_data()
        if self.format == "csv":
            pass
    
        nodes = []
        if self.chunker == "basic":
            nodes = self.basic_chunks(docs)
        if self.chunker == "LLM":
            nodes = self.llm_chunks(docs)
        return nodes
