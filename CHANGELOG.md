# Changelog

## Version 0.0.1 (2023-04-24)

- Initial release of the package.

##  Version 0.0.1 (2023-04-24)
- bugfix

##  Version 0.0.3 (2023-05-02)
- support for langchain 0.0.155 (langchain contains breaking changes... use this version of promptwatch to work with langchain>=0.0.155)
- support for semantical caching

##  Version 0.0.4 (2023-05-03)
- fix typo in CachedChatLLM which caused problems

##  Version 0.0.5 (2023-05-06)
- fixed dismissed llm_output in LlmPrompt metadata
- fixed dismissing the memory parameter in for chat models

##  Version 0.0.6 (2023-05-06)
- fixed potential problems caused by deepcopying of handler in langchain which could lead to breaking the singleton principle
- fix cached llm
- better prompt logging for chat

##  Version 0.0.7 (2023-05-07)
- one more fix for langchain deepcopying and context sharing - fixing not propagating some additional info

##  Version 0.0.8 (2023-05-10)
- fix typing error in caching (caused problems with python 3.10+)
- fixed nesting context problem
- disabled exception on get cache that has not been initialized (reinitializing instead)