"""
Author: Duncan Goudie
License: AGPL-3.0 License
Please contact me directly if you wish to use this software with a different license.

Run PreprocessingPipeline across all pdfs within a folder

"""

import os
import argparse
from datetime import datetime
import json
import re
import uuid

from scripts.preprocess_pdf import PreprocessingPipeline

from scripts.preprocess_pdf import OverallState

def create_pdf_list(document_folder):
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

    pdf_files = [file for file in os.listdir(document_folder) if file.lower().endswith(".pdf")]
    return pdf_files

def create_suffix_list(document_folder, suffix:str='pdf'):
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


def compile_ready_processed_pdfs(folderpath):
    json_filenames = create_suffix_list(document_folder=folderpath, suffix='json')
    json_filepaths = [os.path.join(folderpath, f) for f in json_filenames]

    # Read each json file
    pdf_filepaths = []
    for idx, json_filepath in enumerate(json_filepaths):
        with open(json_filepath) as json_file:
            json_contents = json.load(json_file)

        pdf_filepath = json_contents["metadata"]["original_filepath"]
        pdf_filepaths.append(pdf_filepath)

    return pdf_filepaths

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", help="folderpath to input pdfs", action="store")
    parser.add_argument("-d", help="folderpath to save jsons", action="store")

    # TODO: decide how we want to structure the saves

    args = parser.parse_args()
    input_folderpath = args.i
    save_folderpath = args.d

    original_pdf_filename_list = create_pdf_list(document_folder=input_folderpath)
    original_pdf_filepath_list = [os.path.join(input_folderpath, f) for f in original_pdf_filename_list]
    ready_extracted_pdf_list = compile_ready_processed_pdfs(folderpath=save_folderpath)
    pdf_list = list(set(original_pdf_filepath_list) - set(ready_extracted_pdf_list))

    for idx, pdf_filename in enumerate(pdf_list):
        print(f"Processing document number {idx} of {len(pdf_list)}. Document name: {pdf_filename}")
        pdf_full_filepath = os.path.join(input_folderpath, pdf_filename)

        pp = PreprocessingPipeline()
        successful_doc_load = pp.load_document(pdf_full_filepath)
        if successful_doc_load == False:
            continue
        app = pp.compile()
        state = pp.initialise_state() # TODO: implement this
        result = app.invoke(state)
        #print(result)
        #document_name = result["document_title"] # Doesn't exist in this take home task

        # save the results
        filename = os.path.splitext(os.path.basename(pdf_full_filepath))[0]
        metadata = {"original_filepath": pdf_full_filepath,
                    "filename": filename}
        #document_date = result["document_date"]

        save_extractions(save_folder=save_folderpath,
                         extractions=result,  # TODO: double check that this is correct
                         metadata=metadata)


if __name__ == "__main__":
    main()