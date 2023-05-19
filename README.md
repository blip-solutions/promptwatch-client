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

## Comprehensive Chain Execution Tracking

With PromptWatch.io, you can track all chains, actions, retrieved documents, and more to gain complete visibility into your system. This makes it easy to identify issues with your prompts and quickly fix them for optimal performance.

What sets PromptWatch.io apart is its intuitive and visual interface. You can easily drill down into the chains to find the root cause of any problems and get a clear understanding of what's happening in your system.

![](https://docs.promptwatch.io/assets/images/sessions_optimized.gif)

Read more here:
[Chain tracing documentation](https://docs.promptwatch.io/docs/category/chain-tracing)

## LLM Prompt caching
It is often tha case that some of the prompts are repeated over an over. It is costly and slow. 
With PromptWatch you just wrap your LLM model into our CachedLLM interface and it will automatically reuse previously generated values.

Read more here:
[Prompt caching documentation](https://docs.promptwatch.io/docs/caching)

## LLM Prompt Template Tweaking

Tweaking prompt templates to find the optimal variation can be a time-consuming and challenging process, especially when dealing with multi-stage LLM chains. Fortunately, PromptWatch.io can help simplify the process!

With PromptWatch.io, you can easily experiment with different prompt variants by replaying any given LLM chain with the exact same inputs used in real scenarios. This allows you to fine-tune your prompts until you find the variation that works best for your needs.

![](https://docs.promptwatch.io/assets/images/prompt_templates_optmized.gif)

Read more here:
[Prompt tweaking documentation](https://docs.promptwatch.io/docs/prompt_tweaking)


## Keep Track of Your Prompt Template Changes

Making changes to your prompt templates can be a delicate process, and it's not always easy to know what impact those changes will have on your system. Version control platforms like GIT are great for tracking code changes, but they're not always the best solution for tracking prompt changes.

![](https://docs.promptwatch.io/assets/images/prompt_templates_optmized.gif)

Read more here:
[Prompt template versioning documentation](https://docs.promptwatch.io/docs/prompt_template_versioning)



## Unit testing
Unit tests will help you understand what impact your changes in Prompt templates and your code can have on representative sessions examples.

![](https://docs.promptwatch.io/assets/images/unit_tests.png)
Read more here:
[Unit tests documentation](https://docs.promptwatch.io/docs/category/unit-testing)