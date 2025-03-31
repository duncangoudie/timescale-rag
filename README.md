# timescale-rag
An example RAG and LLM-agent chatbot implementation using Timescale's vector database.

My system consists of 3 stages:
1. Document Preprocessing
2. Vector Database Ingestion
3. LLM-agent Chatbot

The document preprocessing stage is used to extract key pieces of information from the document.

At vector database ingestion, we set the key extractions as vector indexes. This performs better than textbook document chunking since multiple concepts could be contained within a chunk and lead to a neutralised embedding vector. This stage also manages the metadata associated with each vector.

The chatbot uses a ReAct LLM-agent architecture. The tools bound to the LLM are used to query the vector database according to LLM determined criteria from the user prompt; metadata associated with the vector is used for filtering retrieved datapoints. Taking the retrievals from the vector database into account, a final response is handed back to the user.

## Instructions

### Set up environment
Create a docker image, for example
```shell
cd docker
docker build -f Dockerfile-llm-timescale-dev -t [image name] .
```

Then load up an interactive container instance of it
```shell
docker run -it -v /path/to/repo:/rag/ --name timescale_rag [image name] $SHELL
```

### Preprocess Documents
Run the preprocessing script to extract key bits of information from a folder full of pdf files.
```shell
export PYTHONPATH=$PYTHONPATH:/path/to/repo
export OPENAI_API_KEY=""
cd scripts
python3 script_preprocess_pdf_folder.py -i /path/to/repo/data/raw/ -d /path/to/repo/data/extraction/
```
This will take the pdf files inside `data/raw/`, extract some key bits of information from them, and then save it as json files within `data/extraction/`

### Upload to Timescale Vector Database

Set up your Vector Database on Timescale. They will give you a environment variables needed to remotely access your database.

Then you can run the following script to upload your json extraction files.

```shell
export TIMESCALE_SERVICE_URL='postgres://...'
export TIMESCALE_COLLECTION_NAME='rag_demo'
cd scripts
python3 upload_json_extractions_to_timescale.py -i /path/tp/repo/data/extraction/
```

### Run the LLM-Agent Chatbot

Run the chatbot with,
```shell
export TIMESCALE_SERVICE_URL='postgres://...'
export TIMESCALE_COLLECTION_NAME='rag_demo'
export OPENAI_API_KEY=""
cd scripts
python3 prototype_agent_chatbot_timescale.py
```

Example question and answer response from our LLM-agent chatbot,
```
What is your question? How efficient is our block chain technology?
Answer:
To evaluate the efficiency of your blockchain technology, we can analyze several metrics retrieved from various data sources. Here are the key performance indicators:

1. **Supply Chain Efficiency**: 35% improvement noted.
2. **Traffic Efficiency**: 25% improvement reported.
3. **Cost Reduction**: Annual savings amounting to $5 million.
4. **Energy Savings**: A 15% reduction in energy consumption.
5. **Reduced Downtime**: A significant decrease of 50%.
6. **Cost Savings**: Noted to be around 20%, specifics not provided.
7. **Improved Response Time**: 85% of responses are completed in under 5 seconds.
8. **Reduction in Support Costs**: 40% decrease observed.
9. **Higher Engagement**: Customer engagement improved by 45%.
10. **Enhanced Customer Satisfaction**: Increased by 30%.
11. **Extended Equipment Life**: An improvement of 30%, specifics not provided.
12. **Customer Retention**: Increased by 30%.

Overall, these metrics suggest that your blockchain technology is demonstrating significant efficiencies in various areas such as cost savings, response times, and customer engagement.
This result costed the following tokens: {'first_pass_tool_calling_llm': {'output_tokens': 21, 'input_tokens': 123, 'total_tokens': 144}, 'construct_output_response_to_user': {'output_tokens': 244, 'input_tokens': 353, 'total_tokens': 597}}

```

## License

I've licensed this codebase under the GNU General Public License v3.0.

If you would like to use this code under a different license, please reach out to me :)
