"""
Author: Duncan Goudie
License: GNU General Public License v3.0
Please contact me directly if you wish to use this software with a different license.

"""

import json
import os
from typing import List
from tqdm import tqdm

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores.upstash import UpstashVectorStore
from langchain_community.vectorstores.timescalevector import TimescaleVector

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

import argparse
from datetime import datetime



class TimescaleJsonUploader():

    def __init__(self):

        self.SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]
        self.COLLECTION_NAME = os.environ["TIMESCALE_COLLECTION_NAME"] # TODO: make this an environment variable

        # Initialise the VectorStore object
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = TimescaleVector(
            embedding=embeddings,
            service_url=self.SERVICE_URL,
            collection_name=self.COLLECTION_NAME
        )

    def upload_json_to_timescale(self, src_folderpath: str):

        json_filename_list = self._create_suffix_list(document_folder=src_folderpath, suffix="json")
        json_filepaths = [os.path.join(src_folderpath, f) for f in json_filename_list]

        # to test with, let's just stick with 1 file
        #if len(json_filepaths) > 0:
        #    json_filepath = json_filepaths[0]
        #    print(f"Uploading json file {json_filepath}")

        for json_filepath in tqdm(json_filepaths, desc="Uploading json"):
            # load the json into memory
            with open(json_filepath) as json_file:
                json_extractions = json.load(json_file)

            quantitative_extractions_dicts = self._prepare_quantitative_extractions(json_extractions=json_extractions)
            project_extractions_dicts = self._prepare_project_extractions(json_extractions=json_extractions)

            # Perform the upload on each of the dicts
            # What is interesting, is that in the current implementation, it calls the self referenced vectorstore object
            # so that means we don't need to make changes right now. But we should be wary if this implementation changes,
            # this could get broken, so some eventual decoupling would be good
            # Or maybe re-implementing with a choice of vectorstore in the arguement
            self._upload_dicts_to_vectorstore(vectorstore=self.vectorstore, content_list_dict=quantitative_extractions_dicts)
            self._upload_dicts_to_vectorstore(vectorstore=self.vectorstore, content_list_dict=project_extractions_dicts)

    def _create_suffix_list(self, document_folder, suffix: str = 'pdf'):
        """
        List all PDF files in the given folder.

        Args:
            folder_path (str): The path to the folder.

        Returns:
            list: A list of PDF file names.
        """
        if not os.path.isdir(document_folder):
            print(f"Error: {document_folder} is not a valid directory.")
            return []

        pdf_files = [file for file in os.listdir(document_folder) if file.lower().endswith(f".{suffix}")]
        return pdf_files



    def _prepare_quantitative_extractions(self, json_extractions: dict) -> List[dict]:
        """
        Return dict has key-values: "index_embedding"-str, "metadata"-dict
        metadata dict has keys:
        - content_type - "quantitative_extraction"
        - reference_chunk - pointer to chunk number in document_chunks2 in json file. These chunks should be stored somewhere in another database
        - source_document - id of original document it was sourced from (there should be a table which links id to rest of doc details)
        - date
        - date (quarter)
        - ticker
        - company_name

        :param quantitative_extractions:
        :return:
        """
        metadata_dict = self._compile_metadata(json_extractions=json_extractions)
        # Add new metadata specific to this type of vector that applies to all chunks
        metadata_dict["extraction_type"] = "quantitative_extraction"
        quantitative_extractions = json_extractions["extractions"]["quantitative_extractions"]
        #print(f"DEBUG {quantitative_extractions=}")
        quant_extraction_dicts = []
        for chunk_key in quantitative_extractions.keys():
            chunk_id = chunk_key
            chunk_metadata_dict = metadata_dict.copy()
            chunk_metadata_dict["document_chunk_id"] = chunk_id

            chunk_quant_extraction = quantitative_extractions[chunk_id]
            for quant_key in chunk_quant_extraction.keys():
                #quant_key = chunk_quant_key
                # there can be multiple values under a quant_key (this is yet, another dict)
                # solution for now is to combine them into a single string. In most cases, it is 2 parts of the same number
                # TODO: might need to debug and verify this
                #print(f"DEBUG {chunk_quant_extraction=}")
                quant_value = "".join(v for v in chunk_quant_extraction[quant_key])

                # The langchain vectorstore add_doc/add_text method automatically uses page_content as both page_content
                # and embedding index to the entry in the VectorDB. For the quant case, we only want the key part to be
                # indexed. We want the quant value to be retrieved given a key embedding vector search. Reason is that
                # we don't want the numbers to obscure or make vector retrieval less accurate / good.
                quant_metadata_dict = chunk_metadata_dict.copy()
                quant_metadata_dict["quant_value"] = quant_value
                quant_extraction_dict = {"page_content": quant_key,
                                         "metadata": quant_metadata_dict}
                quant_extraction_dicts.append(quant_extraction_dict)

        return quant_extraction_dicts

    def _prepare_project_extractions(self, json_extractions: dict) -> List[dict]:
        metadata_dict = self._compile_metadata(json_extractions=json_extractions)
        # Add new metadata specific to this type of vector that applies to all chunks
        metadata_dict["extraction_type"] = "project_extraction"
        project_extractions = json_extractions["extractions"]["project_extractions"]
        project_extraction_dicts = []
        for chunk_key in project_extractions.keys():
            chunk_id = chunk_key
            chunk_metadata_dict = metadata_dict.copy()
            chunk_metadata_dict["document_chunk_id"] = chunk_id

            chunk_project_extractions = project_extractions[chunk_id]
            for project_description in chunk_project_extractions:
                project_metadata_dict = chunk_metadata_dict.copy() # We don't make any changes here, but it's a good pattern to keep
                project_extraction_dict = {"page_content": project_description,
                                           "metadata": project_metadata_dict}
                project_extraction_dicts.append(project_extraction_dict)

        return project_extraction_dicts

    def _upload_dicts_to_vectorstore(self, vectorstore, content_list_dict: List[dict], alt_key=None):
        """

        :param content_list_dict: Is a list of dicts. The dict must have these 2 keys: page_content and metadata. With values
        of types string and dict respectively.
        :return:
        """
        # exit if either are empty
        if len(content_list_dict) == 0:
            return

        #content_texts, content_metadatas = zip([(c['page_content'], c['metadata']) for c in content_list_dict])
        if alt_key is None:
            content_texts = [c['page_content'] for c in content_list_dict]
        else:
            content_texts = [c[alt_key] for c in content_list_dict]
        content_metadatas = [c['metadata'] for c in content_list_dict]

        # exit if either are empty
        if len(content_texts) == 0 or len(content_metadatas) == 0:
            return

        vectorstore.add_texts(texts=content_texts, metadatas=content_metadatas)

    def _compile_metadata(self, json_extractions: dict):

        document_name = json_extractions["extractions"]["document_name"]

        filename = json_extractions["metadata"]["filename"]
        original_filepath = json_extractions["metadata"]["original_filepath"]

        original_filename = os.path.basename(original_filepath)

        metadata_dict = {"document_name": document_name,
                         "filename": filename,
                         "original_filename": original_filename}

        return metadata_dict

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", help="folderpath to json files", action="store")

    args = parser.parse_args()


    if args.i is not None:

        json_folderpath = args.i

        uploader = TimescaleJsonUploader()
        uploader.upload_json_to_timescale(src_folderpath=json_folderpath)

if __name__ == "__main__":
    main()
