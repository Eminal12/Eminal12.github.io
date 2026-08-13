# Here I am importing all the necessary libraries needed for the entire process of RAG.
# FAISS handles vector similarity search, while the OpenAI and LangChain libraries are
# used to work with Azure OpenAI models. Also, environment variables are loaded from a
# .env file.

from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from doc_text_extraction import extract_text_from_file_markdown
import os
import dotenv
import time
from datetime import datetime

import faiss
import numpy as np

dotenv.load_dotenv(dotenv.find_dotenv())

AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_GPT_4o = os.environ.get("MODEL_GPT_4o")
MODEL_GPT_4o_MINI = os.environ.get("MODEL_GPT_4o_MINI")
API_VERSION = os.environ.get("API_VERSION")
TEXT_EMBEDDING = os.environ.get("TEXT_EMBEDDING")

# This class is in charge of one financial document and all of the steps that go into processing it.
# It extracts documents, makes embeddings, builds an FAISS index, and allows you to run queries based
# on retrieval. When the document is made, it is processed and ready for semantic search. The PDF is
# turned into structured Markdown content, and each page gets an embedding. The vectors are then stored
# in an FAISS index. The GPT model is accessed through Azure OpenAI, which is used later for question-
# answering stage. Then, the embedding model converts text into numerical vector which represent semantic
# meaning. The PDF file is then processed and converted into structural markdown representations of the page
# so that it keeps the alignment of the table. Aftwe, each page of text is transformed into a vector
# representaion so that semantic similarity searches can later be performed. Lastly, a FAISS index is 
# created to store all the page embeddings to allow efficient similarity search.
class FincialDocument:

    def __init__(
        self,
        path: str,
        temp_folder: str,
        openai_key: str = OPENAI_API_KEY,
        verbose: bool = True,
    ):

        # Verbosity and logging
        self.verbose = verbose
        if self.verbose:
            print(f'{datetime.now().strftime("%H:%M:%S")} | Initialising document...')

        # OpenAI Model
        self.model = AzureChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            deployment_name=MODEL_GPT_4o_MINI,
            openai_api_version=API_VERSION,
            openai_api_type="azure",
            temperature=0,
        )

        # OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=TEXT_EMBEDDING,
            model="text-embedding-ada-002",
            openai_api_version=API_VERSION,
            openai_api_key=OPENAI_API_KEY,
            openai_api_type="azure",
            azure_endpoint=AZURE_ENDPOINT,
        )
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=OPENAI_API_KEY,
            api_version=API_VERSION,
        )

        self.path = path
        self.openai_key = openai_key

        texts_extracted = extract_text_from_file_markdown(
            self.path, openai_key, workers=15, temp_folder=temp_folder
        )

        self.texts_search = sorted(
            texts_extracted, key=lambda x: int(x.metadata["page_num"])
        )

        texts_to_embed = [item.page_content for item in self.texts_search]

        embeddings = self.embeddings.embed_documents(texts_to_embed)

        embedding_dim = len(embeddings[0])
        embedding_matrix = np.array(embeddings, dtype='float32')

        self.index = faiss.IndexFlatL2(embedding_dim)

        self.index.add(embedding_matrix)

# Converts the user query into a vector representation by applying the same embedding model to the content pages.
    def embed_query(self, query: str):
        """Embed a query string using the same embedding client."""
        return np.array(self.embeddings.embed_query(query), dtype='float32')

# Finds the most relevant document by comparing embedded query vector with the stored vectors in the FAISS index
    def retrieve_top_k_documents(self, query: str, k: int = 5):
        """Retrieve top k documents most similar to the query."""
        query_embedding = self.embed_query(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        top_docs = [self.texts_search[i] for i in indices[0]]
        return top_docs

# This step retrieves relevant document pages and asks the language model to find the company’s reported revenue.
# The output must follow a structured schema.
    def revenue(self):

        schema = ResponseSchema(
            name="revenue",
            description="The value of the revenue",
        )

        output_parser = StructuredOutputParser.from_response_schemas([schema])

        format_instructions = output_parser.get_format_instructions()

        template = """

        You are a financial analyst assistant specialised in interpreting consolidated financial statements of companies.

        Given the following extracted document content, please provide clear and concise answers to the question below. Focus only on information explicitly stated in the document.

        Total revenue can also be referred to as sales.

        If there is no informarion around revenue return "N/A" as answer.

        {format_instructions}

        Question: {query}
        Answer:
        """

        prompt_template = PromptTemplate(
            template=template,
            partial_variables={"format_instructions": format_instructions},
            input_variables=["query"],
        )

        query = """What is the total revenue?"""

        prompt = prompt_template.format(query=query)

        top_docs = self.retrieve_top_k_documents(query, k=5)

        chain = load_qa_chain(
            llm=self.model,
            chain_type="stuff",
        )

        attempts = 0
        while attempts < 10:
            try:
                output = chain(
                    {
                        "input_documents": top_docs,
                        "question": prompt,
                        "query": query,
                    },
                    return_only_outputs=True,
                )
                output = output["output_text"]

                response_as_dict = output_parser.parse(output)
                break
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed with error: {str(e)}")
                time.sleep(2)
        else:
            raise Exception(f"Code failed after {10} attempts")

        return response_as_dict["revenue"]

    def profit(self):

        schema = ResponseSchema(
            name="profit",
            description="The value of the profit",
        )

        output_parser = StructuredOutputParser.from_response_schemas([schema])

        format_instructions = output_parser.get_format_instructions()

        template = """

        You are a financial analyst assistant specialised in interpreting consolidated financial statements of companies.

        Given the following extracted document content, please provide clear and concise answers to the question below. Focus only on information explicitly stated in the document.

        Profit can also be referred to as net income.

        If there is no informarion around profit or net income return "N/A" as answer.

        {format_instructions}

        Question: {query}
        Answer:
        """

        prompt_template = PromptTemplate(
            template=template,
            partial_variables={"format_instructions": format_instructions},
            input_variables=["query"],
        )

        query = """What is the profit for the reporting period?"""

        prompt = prompt_template.format(query=query)

        top_docs = self.retrieve_top_k_documents(query, k=5)

        chain = load_qa_chain(
            llm=self.model,
            chain_type="stuff",
        )

        attempts = 0
        while attempts < 10:
            try:
                output = chain(
                    {
                        "input_documents": top_docs,
                        "question": prompt,
                        "query": query,
                    },
                    return_only_outputs=True,
                )
                output = output["output_text"]

                response_as_dict = output_parser.parse(output)
                break
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed with error: {str(e)}")
                time.sleep(2)
        else:
            raise Exception(f"Code failed after {10} attempts")

        return response_as_dict["profit"]

    def assets(self):

        schema = ResponseSchema(
            name="assets",
            description="The value of the total assest",
        )

        output_parser = StructuredOutputParser.from_response_schemas([schema])

        format_instructions = output_parser.get_format_instructions()

        template = """

        You are a financial analyst assistant specialised in interpreting consolidated financial statements of companies.

        Given the following extracted document content, please provide clear and concise answers to the question below. Focus only on information explicitly stated in the document.

        If there is no informarion around total assests return "N/A" as answer.

        {format_instructions}

        Question: {query}
        Answer:
        """

        prompt_template = PromptTemplate(
            template=template,
            partial_variables={"format_instructions": format_instructions},
            input_variables=["query"],
        )

        query = """What are the total assets as mentioned in the document?"""

        prompt = prompt_template.format(query=query)

        top_docs = self.retrieve_top_k_documents(query, k=5)

        chain = load_qa_chain(
            llm=self.model,
            chain_type="stuff",
        )

        attempts = 0
        while attempts < 10:
            try:
                output = chain(
                    {
                        "input_documents": top_docs,
                        "question": prompt,
                        "query": query,
                    },
                    return_only_outputs=True,
                )
                output = output["output_text"]

                response_as_dict = output_parser.parse(output)
                break
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed with error: {str(e)}")
                time.sleep(2)
        else:
            raise Exception(f"Code failed after {10} attempts")

        return response_as_dict["assets"]

    def equity(self):

        schema = ResponseSchema(
            name="equity",
            description="The value of the shareholders’ equity",
        )

        output_parser = StructuredOutputParser.from_response_schemas([schema])

        format_instructions = output_parser.get_format_instructions()

        template = """

        You are a financial analyst assistant specialised in interpreting consolidated financial statements of companies.

        Given the following extracted document content, please provide clear and concise answers to the question below. Focus only on information explicitly stated in the document.

        If there is no informarion around shareholders' equity return "N/A" as answer.

        {format_instructions}

        Question: {query}
        Answer:
        """

        prompt_template = PromptTemplate(
            template=template,
            partial_variables={"format_instructions": format_instructions},
            input_variables=["query"],
        )

        query = """What is the shareholders’ equity reported?"""

        prompt = prompt_template.format(query=query)

        top_docs = self.retrieve_top_k_documents(query, k=5)

        chain = load_qa_chain(
            llm=self.model,
            chain_type="stuff",
        )

        attempts = 0
        while attempts < 10:
            try:
                output = chain(
                    {
                        "input_documents": top_docs,
                        "question": prompt,
                        "query": query,
                    },
                    return_only_outputs=True,
                )
                output = output["output_text"]

                response_as_dict = output_parser.parse(output)
                break
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed with error: {str(e)}")
                time.sleep(2)
        else:
            raise Exception(f"Code failed after {10} attempts")

        return response_as_dict["equity"]

    def earnings_per_share(self):

        schema = ResponseSchema(
            name="earnings_per_share",
            description="The value of the earnings per share",
        )

        output_parser = StructuredOutputParser.from_response_schemas([schema])

        format_instructions = output_parser.get_format_instructions()

        template = """

        You are a financial analyst assistant specialised in interpreting consolidated financial statements of companies.

        Given the following extracted document content, please provide clear and concise answers to the question below. Focus only on information explicitly stated in the document.

        If there is no informarion around earnings per share return "N/A" as answer.

        {format_instructions}

        Question: {query}
        Answer:
        """

        prompt_template = PromptTemplate(
            template=template,
            partial_variables={"format_instructions": format_instructions},
            input_variables=["query"],
        )

        query = """What is the earnings per share (EPS)?"""

        prompt = prompt_template.format(query=query)

        top_docs = self.retrieve_top_k_documents(query, k=5)

        chain = load_qa_chain(
            llm=self.model,
            chain_type="stuff",
        )

        attempts = 0
        while attempts < 10:
            try:
                output = chain(
                    {
                        "input_documents": top_docs,
                        "question": prompt,
                        "query": query,
                    },
                    return_only_outputs=True,
                )
                output = output["output_text"]

                response_as_dict = output_parser.parse(output)
                break
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed with error: {str(e)}")
                time.sleep(2)
        else:
            raise Exception(f"Code failed after {10} attempts")

        return response_as_dict["earnings_per_share"]

    def cash_flow(self):

        schema = ResponseSchema(
            name="cash_flow",
            description="The value of the cash flows",
        )

        output_parser = StructuredOutputParser.from_response_schemas([schema])

        format_instructions = output_parser.get_format_instructions()

        template = """

        You are a financial analyst assistant specialised in interpreting consolidated financial statements of companies.

        Given the following extracted document content, please provide clear and concise answers to the question below. Focus only on information explicitly stated in the document.

        cash flows can be values from from operating, investing, and financing activities.

        If there is no informarion around cash flows return "N/A" as answer.

        {format_instructions}

        Question: {query}
        Answer:
        """

        prompt_template = PromptTemplate(
            template=template,
            partial_variables={"format_instructions": format_instructions},
            input_variables=["query"],
        )

        query = """What are the cash flows value?"""

        prompt = prompt_template.format(query=query)

        top_docs = self.retrieve_top_k_documents(query, k=5)

        chain = load_qa_chain(
            llm=self.model,
            chain_type="stuff",
        )

        attempts = 0
        while attempts < 10:
            try:
                output = chain(
                    {
                        "input_documents": top_docs,
                        "question": prompt,
                        "query": query,
                    },
                    return_only_outputs=True,
                )
                output = output["output_text"]

                response_as_dict = output_parser.parse(output)
                break
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed with error: {str(e)}")
                time.sleep(2)
        else:
            raise Exception(f"Code failed after {10} attempts")

        return response_as_dict["cash_flow"]

    def contingent(self):

        schema = ResponseSchema(
            name="contingent",
            description="The value of contingent liabilities",
        )

        output_parser = StructuredOutputParser.from_response_schemas([schema])

        format_instructions = output_parser.get_format_instructions()

        template = """

        You are a financial analyst assistant specialised in interpreting consolidated financial statements of companies.

        Given the following extracted document content, please provide clear and concise answers to the question below. Focus only on information explicitly stated in the document.

        If there is no informarion around contingent liabilities  return "N/A" as answer.

        {format_instructions}

        Question: {query}
        Answer:
        """

        prompt_template = PromptTemplate(
            template=template,
            partial_variables={"format_instructions": format_instructions},
            input_variables=["query"],
        )

        query = """What contingent liabilities are mentioned?"""

        prompt = prompt_template.format(query=query)

        top_docs = self.retrieve_top_k_documents(query, k=5)

        chain = load_qa_chain(
            llm=self.model,
            chain_type="stuff",
        )

        attempts = 0
        while attempts < 10:
            try:
                output = chain(
                    {
                        "input_documents": top_docs,
                        "question": prompt,
                        "query": query,
                    },
                    return_only_outputs=True,
                )
                output = output["output_text"]

                response_as_dict = output_parser.parse(output)
                break
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed with error: {str(e)}")
                time.sleep(2)
        else:
            raise Exception(f"Code failed after {10} attempts")

        return response_as_dict["contingent"]

# This function runs all individual metric extraction functions and returns the results in a single structure.
    def extract_data(self):
        return {"revenue": self.revenue(),
                "profit": self.profit(),
                "assets_and_liabilities": self.assets(),
                "equity": self.equity(),
                "earnings_per_share": self.earnings_per_share(),
                "cash_flow": self.cash_flow(),
                "contingent": self.contingent(),
                }