<h3 align="center">
CoI Agent: Revolutionizing Research in Novel Idea Development with LLM Agents
</h3>

<p align="center">
<a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg"></a>
<a href="https://github.com/DAMO-NLP-SG"><img src="https://img.shields.io/badge/Institution-DAMO-red"></a>
<a><img src="https://img.shields.io/badge/PRs-Welcome-red"></a>
</p>


##  📰 Update
* **[2024.10.12]**  The first version of CoI Agent!


## 🛠️ Requirements and Installation
**Step 1**:
```bash
git clone https://github.com/DAMO-NLP-SG/CoI-Agent.git
cd CoI-Agent
pip install -r requirements.txt
```

**Step 2**:
Install [SciPDF Parser](https://github.com/titipata/scipdf_parser) for PDF parsing.


### API config
set config.yaml to use the OpenAI APIs.
```yaml
# Sementic scholor api, it should be filled
SEMENTIC_SEARCH_API_KEY: ""

# Openai api settings
AZURE_OPENAI_ENDPOINT : ""
AZURE_OPENAI_KEY : ""
AZURE_OPENAI_API_VERSION : ""

OPENAI_API_KEY: ""
OPENAI_BASE_URL: ""

# Gemini api settings
GEMINI_API_KEY: ""

# if not set it will be set to the same as main llm,current only support openai
EMBEDDING_API_KEY: ""
EMBEDDING_API_ENDPOINT: ""
EMBEDDING_MODEL: ""

MAIN_LLM_TYPE: "" # "openai" or "gemini"
MAIN_LLM_MODEL: "" # "gpt-4o" or ...

CHEAP_LLM_TYPE: "" # "openai"  or "gemini"
CHEAP_LLM_MODEL: "" # "gpt-4o" or ...
```

## 🚀 Quick Start
**Step 1**:
**Run grobid**
If you git clone https://github.com/titipata/scipdf_parser.git
```bash
cd scipdf_parser
bash serve_grobid.sh
```

elif you download the grobid
```bash
cd grobid
./gradlew run
```

**Step 2**:
```python
python main.py --topic {your research topic}
```

