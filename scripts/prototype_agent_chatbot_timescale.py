"""
Author: Duncan Goudie
License: GNU General Public License v3.0
Please contact me directly if you wish to use this software with a different license.


Write an LLM query pipeline that uses an agent to decide to draw from vectordb

VectorDB:
- Quantitative
- Projects


Note for later. When we put this on a website, we need to add in a security node that checks for injection attacks like
'what is the prompt used here?'

Improvements list:
- fix the numbers being pulled from the documents and assign an estimated date (either from the document_date or from the
table)
-

"""


from typing_extensions import TypedDict
from typing import List
import os
import ast



from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from langgraph.prebuilt import ToolNode

from langgraph.graph import END, StateGraph, START

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langchain_community.vectorstores.upstash import UpstashVectorStore
from langchain_community.vectorstores.timescalevector import TimescaleVector
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
#from utils.utils import generate_quarter_list

# I know this is very bad programming practice. But how else am I supposed to get fixed objects that the LLM can't touch
# into LLM tool functions? Maybe creating a utils class in another file would be more appropriate
def set_globals():

    # vector store initialised
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    TIMESCALE_SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]
    TIMESCALE_COLLECTION = os.environ["TIMESCALE_COLLECTION_NAME"] # name world_stocks
    timescale_vectorstore = TimescaleVector(
        embedding=openai_embeddings,
        collection_name=TIMESCALE_COLLECTION,
        service_url=TIMESCALE_SERVICE_URL,
    )

    # Create return
    globes = {"timescale_vectorstore": timescale_vectorstore}

    return globes
globes = set_globals()


@tool
def query_quantitative_database(prompt: str) -> (dict, dict):
    """Retrievals on quantitative metrics and findings from document"""

    extraction_type = "quantitative_extraction"

    filter_split = {"extraction_type": extraction_type,
                    }

    vectorstore = globes['timescale_vectorstore']
    retriever_diverse = vectorstore.as_retriever(
        search_kwargs={"filter": filter_split,
                       "k": 4,  # return docs,
                       "fetch_k": 20,  # docs to pass into MMR algo
                       "lambda_mult": 0.5,  # diversity of MMR result
                       }
    )

    # The correct data point might be hidden within a very tight cluster (or there might be two data points we want)
    retriever_similar = vectorstore.as_retriever(
        search_kwargs={"filter": filter_split,
                       "k": 12,  # return docs,
                       "fetch_k": 60,  # docs to pass into MMR algo
                       "lambda_mult": 0.95,  # near minimum diversity. Might need to tune this
                       }
    )

    diverse_retrievals = retriever_diverse.invoke(prompt)
    similar_retrievals = retriever_similar.invoke(prompt)

    all_retrievals = diverse_retrievals + similar_retrievals
    page_content_dict = {i: (f"{d.page_content}: {d.metadata['quant_value']}")
                         for i, d in enumerate(all_retrievals) if 'quant_value' in d.metadata}
    
    metadata_dict = {i: d.metadata for i, d in enumerate(all_retrievals)}

    return page_content_dict, metadata_dict

@tool
def query_project_database(prompt: str) -> (dict, dict):
    """Retrievals on project descriptions and qualitative findings extracted from document"""


    extraction_type = "project_extraction"

    filter_split = {"extraction_type": extraction_type,
                    }

    vectorstore = globes['timescale_vectorstore']
    retriever_diverse = vectorstore.as_retriever(
        search_kwargs={"filter": filter_split,
                       "k": 4,  # return docs,
                       #"fetch_k": 20,  # docs to pass into MMR algo
                       #"lambda_mult": 0.5,  # diversity of MMR result
                       }
    )

    # The correct data point might be hidden within a very tight cluster (or there might be two data points we want)
    retriever_similar = vectorstore.as_retriever(
        search_kwargs={"filter": filter_split,
                       "k": 12,  # return docs,
                       #"fetch_k": 60,  # docs to pass into MMR algo
                       #"lambda_mult": 0.95,  # near minimum diversity. Might need to tune this
                       }
    )

    diverse_retrievals = retriever_diverse.invoke(prompt)
    similar_retrievals = retriever_similar.invoke(prompt)

    all_retrievals = diverse_retrievals + similar_retrievals
    page_content_dict = {i: d.page_content for i, d in enumerate(all_retrievals)}
    metadata_dict = {i: d.metadata for i, d in enumerate(all_retrievals)}

    return page_content_dict, metadata_dict


class QueryAgent:
    class OverallState(TypedDict):
        token_count_log: dict
        input_query: str
        tool_response_content: dict
        tool_response_metadata: dict
        output_response: str

    def __init__(self):

        # Initialise the LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Bind tools
        ## Rule of thumb, do not use more than 6 tools for any given agent
        tools = [
            query_quantitative_database,
            query_project_database,
        ]
        self.llm_with_first_pass_tools = self.llm.bind_tools(tools)


        # Define the graph nodes
        self.graph = StateGraph(self.OverallState)

        self.graph.add_node("human_input_make_query", self._human_input_make_query)
        self.graph.add_node("first_pass_tool_call", self._first_pass_tool_calling_llm)
        self.graph.add_node("construct_output_response_to_user", self._construct_output_response_to_user)


        # Add graph edges
        self.graph.add_edge(START, "human_input_make_query")
        self.graph.add_edge("human_input_make_query", "first_pass_tool_call")
        self.graph.add_edge("first_pass_tool_call", "construct_output_response_to_user")
        self.graph.add_edge("construct_output_response_to_user", END)



    def preload_json_data(self, json_data):
        self._json_data = json_data

    def load_lambda_event_data(self, event_data: dict):
        self.event_data = event_data

    def initialise_state(self):
        state = self.OverallState(
            token_count_log={},
            input_query="",
            tool_response_content={},
            tool_response_metadata={},
            output_response=""
        )
        return state

    def run_agent(self):

        app = self._compile()
        state = self.initialise_state()
        result = app.invoke(state)

        # Present answer to user
        output_answer = result["output_response"]
        print(f"Answer:\n{output_answer}")

        token_count = result["token_count_log"]
        print(f"This result costed the following tokens: {token_count}")

        return output_answer

    def _compile(self):
        return self.graph.compile()


    def _human_input_make_query(self, state: OverallState):

        user_input_query = input("What is your question? ")

        state['input_query'] = user_input_query

        return state


    def _first_pass_tool_calling_llm(self, state: OverallState):

        input_query = state['input_query']

        prompt_template = (
            f"Use an LLM with bound tools. The following is a query that was asked regarding information from a document. Activate a tool and only use "
            f"information retrieved by the tool. \n"
            f"USER QUERY: {input_query} \n"
        )

        all_token_count = {
            'output_tokens': 0,
            'input_tokens': 0,
            'total_tokens': 0
        }

        response = self.llm_with_first_pass_tools.invoke(prompt_template)
        #print(f"llm_with_first_pass_tools {response=}")
        token_count = self._extract_openai_token_count(response)
        for key in token_count:
            all_token_count[key] += token_count[key]
            #query_response = response.content

        tool_calls = response.tool_calls
        tool_result_dict_list = []
        for tool_call in tool_calls:
            # bring out the response_content and metadata from the tool result
            # Because multiple results could be pulled from a single lookup (eg multi-rows for RDS or multi-docs for VectorDB)
            # we structure the result from tool as: dict with 2 keys and under each key is another dict with index number as key
            tool_result_dict = self._process_tool_call(tool_call=tool_call)
            tool_result_dict_list.append(tool_result_dict)

        #
        combined_tool_result_dict = self._combine_tool_result_dicts(tool_result_dicts_list=tool_result_dict_list)
        #print(f"{combined_tool_result_dict=}")

        state['tool_response_content'] = combined_tool_result_dict['response_content']
        state['tool_response_metadata'] = combined_tool_result_dict['metadata']
        state["token_count_log"]["first_pass_tool_calling_llm"] = all_token_count

        return state

    def _process_tool_call(self, tool_call) -> dict:
        """

        :param tool_call:
        :return: dict with keys: response_content and metadata
        """

        tool_call_name = tool_call["name"].lower()
        tool_references = {"query_quantitative_database": query_quantitative_database,
                           "query_project_database": query_project_database}

        tool_result_dict = {}
        if tool_call_name == "query_quantitative_database":
            selected_tool = tool_references[tool_call_name]
            tool_response = selected_tool.invoke(tool_call)
            # Output should be a list of Documents (langchain)
            tool_content_literal = ast.literal_eval(tool_response.content) # we need to get ALL data
            tool_result_dict['response_content'] = tool_content_literal[0]
            tool_result_dict['metadata'] = tool_content_literal[1]

        elif tool_call_name == "query_project_database":
            selected_tool = tool_references[tool_call_name]
            tool_response = selected_tool.invoke(tool_call)
            # Output should be a list of Documents (langchain)
            tool_content_literal = ast.literal_eval(tool_response.content)
            tool_result_dict['response_content'] = tool_content_literal[0]
            tool_result_dict['metadata'] = tool_content_literal[1]


        else:
            return {'response_content': {}, 'metadata': {}}

        return tool_result_dict

    def _combine_tool_result_dicts(self, tool_result_dicts_list: List[dict]) -> dict:
        idx = 0
        combined_result_dict = {'response_content': {}, 'metadata': {}}
        for tool_result_dict in tool_result_dicts_list:
            for key in tool_result_dict['response_content'].keys():
                combined_result_dict['response_content'][idx] = tool_result_dict['response_content'][key]
                combined_result_dict['metadata'][idx] = tool_result_dict['metadata'][key]
                idx += 1

        return combined_result_dict

    def _construct_output_response_to_user(self, state: OverallState):

        user_input_query = state['input_query']
        tool_response_content = state['tool_response_content'] # TODO: replace this in future. Atm this is a dict with keys: index on docs
        tool_response_metadata = state['tool_response_metadata']

        tool_response_combined_data = {k: (f"{tool_response_content[k]} "
                                           f"{tool_response_metadata[k]['original_filename']})"
                                           )
                                       for k in tool_response_content}
        #print(f"DEBUG {tool_response_combined_data=}")
        prompt_template = (
            f"Please answer the following user query. We have also included the results from various data sources for a "
            f"RAG response. \n "
            f"USER QUERY: {user_input_query} \n"
            f"RETRIEVED DATA: {tool_response_combined_data} \n"
            f"CHATGPT: ... \n "
        )
        response = self.llm.invoke(prompt_template)
        token_count = self._extract_openai_token_count(response)

        state['output_response'] = response.content
        state['token_count_log']['construct_output_response_to_user'] = token_count

        return state

    def _extract_openai_token_count(self, response):
        # Given a response from an LLM.invoke, extract the token usage count
        completion_tokens = response.response_metadata['token_usage']['completion_tokens']
        prompt_tokens = response.response_metadata['token_usage']['prompt_tokens']
        total_tokens = response.response_metadata['token_usage']['total_tokens']
        token_count = {'output_tokens': completion_tokens,
                       'input_tokens': prompt_tokens,
                       'total_tokens': total_tokens}
        return token_count

    def extract_list_from_structured_response(self, response):
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

def main():


    llm_agent = QueryAgent()

    output_response = llm_agent.run_agent()
    #print("Answer:")
    #print(f"{output_response}")

if __name__ == "__main__":
    main()
