"""
Author: Duncan Goudie
License: AGPL-3.0 License
Please contact me directly if you wish to use this software with a different license.
"""



import argparse

import operator
import os
from typing import Annotated
from typing_extensions import TypedDict
from datetime import datetime
import json
from tqdm import tqdm
import ast
import re


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#from langgraph.types import Send
from langgraph.graph import END, StateGraph, START

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from langchain_core.tools import InjectedToolArg, tool
from typing_extensions import Annotated


# TODO: we need to initialise these
# TODO: save this as a json for later exploration (minus the contents)
class OverallState(TypedDict):
    contents: list
    document_chunks2: list
    token_count_log: dict
    document_title: str
    project_extractions: dict
    quantitative_extractions: dict


class PreprocessingPipeline:

    def __init__(self, input_params: dict = {}) -> None:


        self.llm_model_name = "gpt-4o-mini-2024-07-18"
        self.llm = ChatOpenAI(model=self.llm_model_name, temperature=0.2)


        # define the graph
        self.graph = StateGraph(OverallState)

        ## Nodes
        self.graph.add_node("setup_memory", self._setup_memory)
        self.graph.add_node("extract_title", self._extract_document_title)
        self.graph.add_node("extract_information_from_chunks", self._extract_information_from_chunks)

        ## Edges
        self.graph.add_edge(START, "setup_memory")
        self.graph.add_edge("setup_memory", "extract_title")
        self.graph.add_edge("extract_title", "extract_information_from_chunks")
        self.graph.add_edge("extract_information_from_chunks", END)



    def load_document(self, pdf_filepath: str):
        # Load doc
        loader = PyPDFLoader(pdf_filepath)
        self.document = loader.load()

        # Split doc
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50)
        self.document_chunks1 = text_splitter.split_documents(self.document)

        text_splitter2 = RecursiveCharacterTextSplitter(
            chunk_size=2040,
            chunk_overlap=100)
        self.document_chunks2 = text_splitter2.split_documents(self.document)
        print(f"{len(self.document_chunks1)=} {len(self.document_chunks2)=}")

        if len(self.document_chunks1) == 0 or len(self.document_chunks2) == 0:
            return False
        else:
            return True

    def initialise_state(self):
        state = OverallState(
            contents=[],
            document_chunks2={},
            token_count_log={},
            document_title="",
            project_extractions={},
            quantitative_extractions={},
        )
        return state

    def _setup_memory(self, state: OverallState):
        for idx, chunk in enumerate(self.document_chunks2):
            state['document_chunks2'][idx] = chunk

        #print(f"DEBUG {state['document_chunks2']=}")

        return state

    def _nothing_node(self, state: OverallState):

        return state

    def _extract_document_title(self, state: OverallState):
        early_select_doc_chunks = self._publish_early_document_chunks(self.document_chunks1, target_number=3)
        if len(early_select_doc_chunks) > 0:
            document_metadata = self.document_chunks1[0].metadata
        else:
            document_metadata = None
        prompt_template = (
            f"The following is from a financial document from company investor relations. \n"
            f"Please extract the title of this document. "
            f"Usually the first few words or lines contain the document title. \n"
            f"DOCUMENT CONTENTS: {early_select_doc_chunks} \n"  # This is assuming chunk size of 500
            f"DOCUMENT METADATA: {document_metadata} \n"
            f"CHATGPT: please only respond with a category from the list. No chat."
        )

        response = self.llm.invoke(prompt_template)
        token_count = self._extract_openai_token_count(response)
        document_title = response.content

        state["document_title"] = document_title
        state["token_count_log"]["document_title"] = token_count
        return state

    def _extract_information_from_chunks(self, state: OverallState):
        """
        Scan through every chunk in the document, deciding which are the best tools for extracting information from it.
        Just run through all tools anyway.
        """
        print(f"Started _extract_information_from_chunks at time {datetime.now()}")
        chunks = self.document_chunks2
        all_token_count = {
            'output_tokens': 0,
            'input_tokens': 0,
            'total_tokens': 0
        }
        # idx corresponds to index of chunk list
        for idx, chunk in enumerate(tqdm(chunks, desc="Extracting information from chunk")):
            try:
                # run the extract_process_informatino method
                extraction_project_dict, token_count = self._sub_chunk_extract_project_information(document_chunk=chunk)
                for key in token_count:
                    all_token_count[key] += token_count[key]
                state["project_extractions"][idx] = extraction_project_dict

                # run the _sub_chunk_extract_reference_document method
                extraction_ref_doc_dict, token_count = self._sub_chunk_extract_quantitative_information(document_chunk=chunk)
                for key in token_count:
                    all_token_count[key] += token_count[key]
                state["quantitative_extractions"][idx] = extraction_ref_doc_dict


            except Exception as e:
                print(f"Error found. {e=}, {chunk=}")

        state["token_count_log"]["extract_information_from_chunks"] = all_token_count
        print(f"Finished _extract_information_from_chunks at time {datetime.now()}")

        return state

    def _sub_chunk_extract_quantitative_information(self, document_chunk) -> (list, dict):
        """
        Examine the document chunk and extract information regarding projects the company is working on.
        """
        prompt_template = (
            f"The following is a chunk from a document regarding projects. "
            f"Please search this chunk for quantitative data. Extract the name and value. If there is an associated date "
            f"with the value, please extract that too. \n"
            f"DOCUMENT CHUNK: {document_chunk.page_content} \n "  # This is assuming chunk size of 1000
            f"CHATGPT: please structure the response with key-value pairs, separated by a ':'. If there is an associated "
            f"date (or quarter or month or year), please put it in brackets next to the value. For example: \n"
            f"metric1: $123456 (2022Q2)\n"
            f"metric2: 111222 kWh\n"
            f"If nothing is found you can refrain from writing anything."
        )

        response = self.llm.invoke(prompt_template)
        token_count = self._extract_openai_token_count(response)

        extraction_list = self._extract_list_from_structured_response(response=response)

        return extraction_list, token_count

    def _sub_chunk_extract_project_information(self, document_chunk) -> (list, dict):
        """
        Examine the document chunk and extract information regarding projects the company is working on.
        """
        prompt_template = (
            f"The following is a chunk from a document. "
            f"Please search this chunk for information regarding projects.\n"
            f"DOCUMENT CHUNK: {document_chunk.page_content} \n "  # This is assuming chunk size of 1000
            f"CHATGPT: Please structure the response as a list of found items, for example:\n"
            f"-[description and context of project 1]\n"
            f"-[description, details and context of project 2]\n"
            f"If no projects are found, you should write nothing."
        )

        response = self.llm.invoke(prompt_template)
        token_count = self._extract_openai_token_count(response)

        extraction_list = self.extract_list_from_structured_list_response(response=response)

        # Sometimes "no projects were found" or similar are placed inside the list
        # TODO: test this
        delete_phrases = [
            "No specific projects",
            "No projects are found",
            "No projects were found",
            "No projects"
        ]
        cleaned_extraction_list = [extraction for extraction in extraction_list if not any(phrase in extraction for phrase in delete_phrases)]

        return cleaned_extraction_list, token_count


    def compile(self):
        return self.graph.compile()

    def _extract_openai_token_count(self, response):
        # Given a response from an LLM.invoke, extract the token usage count
        completion_tokens = response.response_metadata['token_usage']['completion_tokens']
        prompt_tokens = response.response_metadata['token_usage']['prompt_tokens']
        total_tokens = response.response_metadata['token_usage']['total_tokens']
        token_count = {'output_tokens': completion_tokens,
                       'input_tokens': prompt_tokens,
                       'total_tokens': total_tokens}
        return token_count

    def _clean_response_category(self, response, category_list: list):
        # Expect a single single category string from LLM. Eg annual_report
        # But we need to clean it just in case

        category = response.content
        if '`' in category:
            category = category.replace('`', '')
        elif '"' in category:
            category = category.replace('"', '')
        elif "'" in category:
            category = category.replace("'", '')

        for c in category_list:
            if c in category:
                return c

        if category not in category_list:
            return None

        return category

    def _clean_response_name(self, response):
        # Expect a single single category string from LLM. Eg annual_report
        # But we need to clean it just in case

        name = response.content
        if '`' in name:
            name = name.replace('`', '')
        elif '"' in name:
            name = name.replace('"', '')
        elif "'" in name:
            name = name.replace("'", '')

        return name

    def _check_response_valid_date(self, response):
        """Checks if a string is a valid date in the given format."""
        format = "%Y-%m-%d"
        date_str = response.content
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False

    def extract_list_from_structured_list_response(self, response):
        """
        Extract the list of items written under each heading. Example response:
        JARGON: jargon1, jargon2
        TOPIC: topic1, topic2
        KPI: none
        The heading list is given
        """
        # Split the input string into lines
        input_string = response.content
        lines = input_string.split('\n')

        # Initialize a dictionary to hold the results
        result = []

        # Iterate over each line
        for line in lines:
            # Split the line into a key and value
            if '-' in line:
                key, value = line.split('-', 1)

                # Convert the value to a list, splitting by commas
                if value.lower() != 'none':  # Handle the "none" case
                    result.append(value)

        return result

    def _extract_list_from_structured_response(self, response):
        """
        Extract the list of items written under each heading. Example response:
        JARGON: jargon1, jargon2
        TOPIC: topic1, topic2
        KPI: none
        The heading list is given
        """
        # Split the input string into lines
        input_string = response.content
        lines = input_string.split('\n')

        # Initialize a dictionary to hold the results
        result = {}

        # Iterate over each line
        for line in lines:
            # Split the line into a key and value
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()  # Remove extra spaces
                value = value.strip()  # Remove extra spaces

                # Convert the value to a list, splitting by commas
                if value.lower() != 'none':  # Handle the "none" case
                    result[key] = [item.strip() for item in value.split(',')]
                else:
                    result[key] = []

        return result

    def extract_numbered_dict_from_numbered_list_response(self, response) -> dict:
        """
        Extract the numbered list of items written under each heading. Example response:
        1. abc
        2. 456
        3. xyz
        Return it as a dict where the key is the number and the value is the text beside it.
        """
        text = response.content
        numbered_list = {}
        lines = text.strip().split("\n")
        numbered_list_pattern = re.compile(r"^(\d+)\.\s*(.+)")

        for line in lines:
            match = numbered_list_pattern.match(line.strip())
            if match:
                number, item = match.groups()
                numbered_list[int(number)] = item

        return numbered_list

    def _binary_filter_extracted_item_with_llm(self, item_str, prompt_criteria):
        prompt_template = (
            f"The following string was extracted using an LLM, '{item_str}'.\n"
            f"Please check if it conforms to the following criteria: {prompt_criteria}\n"
            f"If it does, return TRUE. If it does not, return FALSE. "
            f"CHATGPT: please only respond with TRUE or FALSE."
        )
        response = self.llm.invoke(prompt_template)
        token_count = self._extract_openai_token_count(response)
        binary_output = self._clean_response_category(response=response, category_list=['TRUE', 'FALSE'])
        if "true" in binary_output or "True" in binary_output or "TRUE" in binary_output:
            return True, token_count
        else:
            return False, token_count

    def _pick_unique_instances_with_llm(self, item_list: list):
        prompt_template = (
            f"Pick out unique instances from the following list:\n"
            f"{','.join(item_list)}\n"
            f"CHATGPT: please structure the response with a list of found items under the heading: \n"
            f"ITEM_LIST: ...\n"
            f"If there are no items are under the heading, just write NONE under it."
        )
        response = self.llm.invoke(prompt_template)
        token_count = self._extract_openai_token_count(response)
        unique_item_dict = self._extract_list_from_structured_response(response)
        unique_item_list = unique_item_dict['ITEM_LIST']
        return unique_item_list, token_count





    def _is_valid_date_format(self, date_str: str) -> bool:
        """
        Check if the input string is in the format 'YYYY-MM-DD'.

        Args:
            date_str (str): The date string to check.

        Returns:
            bool: True if the date is valid and in the correct format, False otherwise.
        """
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False


    def _publish_early_document_chunks(self, document_chunks, target_number=2):
        if len(document_chunks) >= target_number:
            #print(f"DEBUG {document_chunks[0]=}")
            select_chunks = ""
            for i in range(target_number):
                select_chunks += f"{document_chunks[i].page_content}"

        elif len(document_chunks) == 0:
            select_chunks = " No document chunks were found... "
        elif len(document_chunks) < target_number:
            select_chunks = ""
            for i in range(len(document_chunks)-1):
                select_chunks += f"{document_chunks[i].page_content}"
        else:
            select_chunks = f"{document_chunks[0].page_content}"

        return select_chunks


def save_extractions(save_folder: str, extractions: OverallState, metadata: dict):
    """
    token_count_log: dict
    document_type: str
    document_date: str # TODO: change to datetime object?
    document_summary: str
    business_items: dict
    """

    filename = metadata["filename"]

    document_name = extractions["document_title"]
    document_chunks2 = [{"metadata": extractions["document_chunks2"][k].metadata,
                         "page_content": extractions["document_chunks2"][k].page_content} for k in extractions["document_chunks2"]]
    project_extractions = extractions["project_extractions"]
    quantitative_extractions = extractions["quantitative_extractions"]

    token_count_log = extractions["token_count_log"]

    extractions = {
        "document_name": document_name,
        "document_chunks2": document_chunks2,
        "project_extractions": project_extractions,
        "quantitative_extractions": quantitative_extractions,
        "token_count_log": token_count_log,
    }

    json_contents = {
        "metadata": metadata,
        "extractions": extractions
    }


    save_filename = f"extractions_{filename}.json"
    def replace_all_whitespace(text: str) -> str:
        return re.sub(r"\s+", "_", text)
    def sanitize_filename(filename: str) -> str:
        return filename.replace("/", "_")

    save_filename = sanitize_filename(save_filename)
    save_filename = replace_all_whitespace(save_filename)
    save_filepath = os.path.join(save_folder, save_filename)
    # Save to a JSON file
    with open(save_filepath, "w") as json_file:
        json.dump(json_contents, json_file, indent=2)




