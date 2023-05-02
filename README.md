# PromptWatch.io ... session tracking for LangChain 

It enables you to:
- track all the chain executions
- track LLM Prompts and **re-play the LLM runs** with the same input parameters and model settings to tweak your prompt template
- track your costs per **project** and per **tenant** (your customer)

## Installation 
```bash
pip install promptwatch
```

## Basic usage

In order to enable session tracking  wrap you chain executions in PromptWatch block

```python

from langchain import OpenAI, LLMChain, PromptTemplate
from promptwatch import PromptWatch

prompt_template = PromptTemplate.from_template("Finish this sentence {input}")
my_chain = LLMChain(llm=OpenAI(), prompt=prompt_template)

with PromptWatch(api_key="<your-api-key>") as pw:
    my_chain("The quick brown fox jumped over")

```

Here you can get your API key: http://www.promptwatch.io/get-api-key (no registration needed)

You can set it directly into `PromptWatch` constructor, or set is as an *ENV variable* `PROMPTWATCH_API_KEY`

### Project and Tenant costs tracking

You can assign a **project** and **tenant** id to your session by setting the constructor parameter:

This will allow you to track costs per OpenAI request per customer and as well as your dev project.

```python

...

with PromptWatch(tracking_project="my-project", tracking_tenant="my-tenant",) as pw:
    my_chain("The quick brown fox jumped over")

```
### What is being tracked

PromptWatch tracks all the details that LangChain exposes via its tracking "API" and more.

👉 Chain execution inputs, outputs, execution time

👉 Tools input output

👉 **retrieved documents from retrieval vector DB**

👉 Details about LLM runs like:

  - final prompt text
  - generated text
  - execution details like model, temperature, etc. (everything you need to re-run the prompt with the same exact setup)
  - total used tokens
  - **costs (based on OpenAI price list per model)**
  - **prompt template and its parameters**
  
 

### Custom logging

PromptWatch tracks quite extensively standard LangChain tools, but if you have some custom code you'd like to track you can do so.

```python
...
with PromptWatch(api_key=invalid_api_key):
    PromptWatch.log_activity(Question(text="What did the president say about Ketanji Brown Jackson"))
    PromptWatch.log("my arbitrary log message")
```

All the logs are associated with to opened session. You can't log outside the session.

```python
...
with PromptWatch(api_key=invalid_api_key):
    PromptWatch.log_activity(Question(text="What did the president say about Ketanji Brown Jackson"))
    #end of session   
PromptWatch.log("this will raise an exception!")
```


## Prompt template tracking

You can register any LangChain prompt template for detailed monitoring

```python
from promptwatch import PromptWatch, register_prompt_template
from langchain import OpenAI, LLMChain, PromptTemplate

prompt_template = PromptTemplate.from_template("Finish this sentence {input}")
my_chain = LLMChain(llm=OpenAI(), prompt=prompt_template)

register_prompt_template("your_template_name",prompt_template) 

with PromptWatch() as pw:
    
    #execute the chain
    my_chain("The quick brown fox jumped over")

```

This will allow you to associate the prompt template with a custom name (and function) and track it independently... 

Any change of that template text will cause an automatic version change (with automatic version number increment)

**Warning**
The registration just assigns the template a custom name and is only done when the LLM actually executes the LLM prompt with that template. Therefore is has no additional performance costs, on the contrary it can even speed up the execution a bit if the same template is used multiple times.


