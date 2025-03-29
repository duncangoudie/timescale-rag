# timescale-rag
An example RAG and LLM-agent chatbot implementation using Timescale's vector database.

TODO: draw out the system

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

## License

I've licensed this codebase under the GNU General Public License v3.0.

If you would like to use this code under a different license, please reach out to me :)