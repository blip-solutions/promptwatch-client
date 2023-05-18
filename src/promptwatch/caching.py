from typing import List, Callable, Dict, Union, Tuple, TYPE_CHECKING
from uuid import uuid5, NAMESPACE_DNS, UUID
from abc import ABC, abstractmethod
import tiktoken
import logging

if TYPE_CHECKING:
    from .promptwatch_context import PromptWatch

DEFAULT_CACHE_KEY="default"

class CacheResult:
    def __init__(self, cache_namespace_key, prompt, embedding, result:str=None, metadata:Dict[str,Union[str,int]]=None):
        self.prompt = prompt
        self.embedding = embedding
        self.cache_namespace_key = cache_namespace_key
        self.result = result
        self.metadata=metadata
        
    
    def __bool__(self):
        return bool(self.result)
    

class PromptWatchCache:
    def __init__(self, cache_namespace_key, implementation:"CacheImplBase", embed_func:Callable[[str],List[float]], token_limit, similarity_limit:float=0.97):
        self.cache_namespace_key = cache_namespace_key
        self.implementation:CacheImplBase= implementation
        self.embed_func = embed_func
        self.similarity_limit=similarity_limit
        if isinstance(self.embed_func, EmbeddingProviderBase) and not token_limit:
            token_limit=self.embed_func.token_limit

        self.token_limit=token_limit
        self.logger = logging.getLogger(__name__)
        
        

    def test_token_limit(self, prompt:str)->bool:
         # assuming that the average token length is way over 2 chars, it's safe to assume that if the prompt is half the token limit, we the real limit won't be reached
        if len(prompt)/2 <= self.token_limit:
            return True
        elif num_tokens_from_string(prompt) <= self.token_limit:
            return True
        else:
            return False
        

    def get(self, prompt:str)->Union[CacheResult,None]:
        """ returns cached str if found, NotFoundHandle if not found, None if prompt is too long to be cached
        
        ## Example usage:
        
        ```
        cache_result = cache.get("foo")
        if not cache_result:
            result = llm.run("foo")
            cache.add(cached_result,result)
        else:
            return cache_result.result
        ```
        """
        try:
            if self.test_token_limit(prompt):
                if isinstance(self.embed_func, EmbeddingProviderBase) and self.embed_func.should_include_token_usage:
                    prompt_embedding, token_usage = self.embed_func(prompt, True)
                    metadata={
                        "token_usage":token_usage
                    }
                else:
                    prompt_embedding = self.embed_func(prompt)
                    metadata={}
                result, similarity = self.implementation.get(prompt_embedding,self.similarity_limit) or (None,None)
                if result:
                    metadata["similarity"]=similarity
                
                return CacheResult(self.cache_namespace_key, prompt, prompt_embedding, result=result, metadata=(metadata if result else None))
                
            else:
                self.logger.warning(f"Prompt {prompt} is too long to be cached. It will be ignored")
                return None
        except Exception as e:
            self.logger.warn(f"Skipping cache due to Error: {e}")
    

    def add(self, not_found_handle: CacheResult, result:str):
        # not_found_handle could be None (too long prompt), in which case we ignore it
        if isinstance(not_found_handle,CacheResult):
            try:
                if not_found_handle.cache_namespace_key != self.cache_namespace_key:
                    raise ValueError("PromptWatchCache.add called with a not_found_handle that does not belong to this cache")
                prompt_hash = uuid5(NAMESPACE_DNS, not_found_handle.prompt)
                self.implementation.add(prompt_hash, not_found_handle.embedding, result)
            except Exception as e:
                self.logger.warn(f"Failed storing prompt into cache {e}")
        
    

    def clear(self):
        self.implementation.clear()

class EmbeddingProviderBase:

    @property
    def token_limit(self)->int:
        raise NotImplementedError()
    
    @property
    def should_include_token_usage(self)->bool:
        return False

    @abstractmethod
    def __call__(self, prompt:str, include_token_usage:bool=False) -> Union[List[float], Tuple[List[float],int]]:
        pass
        

class OpenAIEmbeddingProviderBase(EmbeddingProviderBase):
    def __init__(self, model_name="text-embedding-ada-002", token_limit=8191):
        self._token_limit = token_limit
        self.model_name = model_name
        try: 
            import openai
                
       
            def embed(prompt:str)->List[float]:
                # replace newlines, which can negatively affect performance.
                prompt = prompt.replace("\n", " ")
                engine=model_name
                return openai.Embedding.create(input=[prompt], engine=engine)
            
            self.create_embeddings = embed

        except ImportError:
                raise Exception("Unable to import default embeddings provider (OpenAI). Please install openai (pip install openai) or provide either an embed_func")

    
    @property
    def token_limit(self)->int:
        return self._token_limit
    
    @property
    def should_include_token_usage(self)->bool:
        return True
    
    def __call__(self, prompt: str, include_token_usage: bool = False) :
        res = self.create_embeddings(prompt)
        if include_token_usage:
            return res["data"][0]["embedding"], res["usage"]["total_tokens"]
        else:
            res["data"][0]["embedding"]


class PromptWatchCacheManager:

    def __init__(self, promptwatch_context):
        self.promptwatch_context:PromptWatch = promptwatch_context
        self.caches:Dict[str, PromptWatchCache] = {}
        

    def init_cache(self, cache_namespace_key:str=None,  embed_func:Callable[[str],List[float]]=None, token_limit=None, similarity_limit=0.97, local:bool=False)->None:
        if not cache_namespace_key:
            cache_namespace_key = DEFAULT_CACHE_KEY
        if cache_namespace_key in self.caches:
            return
        if not embed_func:
            
            try: 
                embed_func = OpenAIEmbeddingProviderBase()
            except ImportError:
                raise Exception("Unable to import default embeddings provider (OpenAI). Please install openai (pip install openai) or provide either an embed_func")
        elif not isinstance(embed_func, EmbeddingProviderBase) and not token_limit:
            msg = "WARNING: Custom embedding provider set, but no token limit specified for CachedLLM. Using default of 512."
            print("\033[33m" +msg + "\033[0m")
            token_limit=512

        if local:
            self.caches[cache_namespace_key] = PromptWatchCache(cache_namespace_key, LocalImpl(self.promptwatch_context,cache_namespace_key, embed_func), embed_func, token_limit=token_limit, similarity_limit=similarity_limit)
        else:
            self.caches[cache_namespace_key] = PromptWatchCache(cache_namespace_key, RemoteImpl(self.promptwatch_context, cache_namespace_key, embed_func), embed_func, token_limit=token_limit, similarity_limit=similarity_limit)

    
    def get_cache(self, cache_namespace_key=None)->PromptWatchCache:
        return self.get_or_init_cache(cache_namespace_key=cache_namespace_key)
        # cache_namespace_key=cache_namespace_key or DEFAULT_CACHE_KEY
        # if not cache_namespace_key in self.caches:
        #     raise Exception(f"Cache {cache_namespace_key} not initialized")
        # return self.caches[cache_namespace_key]
    
    def get_or_init_cache(self, cache_namespace_key:str=None,  embed_func:Callable[[str],List[float]]=None, token_limit=None, similarity_limit=0.97, local:bool=False)->PromptWatchCache:
        cache_namespace_key=cache_namespace_key or DEFAULT_CACHE_KEY
        if not cache_namespace_key in self.caches:
            self.init_cache(cache_namespace_key=cache_namespace_key,  embed_func=embed_func, token_limit=token_limit,  similarity_limit=similarity_limit, local=local)
        return self.caches[cache_namespace_key]


class CacheImplBase(ABC):

    def __init__(self, promptwatch_context, cache_namespace_key:str =None) -> None:
        self.cache_namespace_key = cache_namespace_key

    @abstractmethod
    def get(self, prompt_embedding:List[float], similarity_limit:float = 0.97)->  Tuple[str, float]:
        pass
    
    @abstractmethod
    def add(self, prompt_hash:UUID, prompt_embedding:List[float],result:str):
        pass

    @abstractmethod
    def clear(self):
        pass
    


class LocalImpl(CacheImplBase):
    def __init__(self, promptwatch_context, cache_namespace_key:str =None,embed_func=None ) -> None:
        try:
            from langchain.vectorstores.chroma import Chroma
            import chromadb
            from chromadb.config import Settings
            from chromadb.api.local import LocalAPI
            from chromadb.api.local import Collection
            
            self.client:LocalAPI = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=f".promptwatch_cache/{cache_namespace_key}"))
            collection = self.client.get_or_create_collection(name=cache_namespace_key or DEFAULT_CACHE_KEY, embedding_function=embed_func or (lambda x:[]))
            
            self.collection:Collection = collection
            self.has_data = self.collection.count()>0
            
            
        except ImportError:
            raise ImportError("chromadb is required for local cache... please use 'pip install chromadb'")
    

    def get(self, prompt_embedding:List[float], similarity_limit:float = 0.97)-> Tuple[str, float]:

        if self.has_data:
            results =  self.collection.query(query_embeddings=[prompt_embedding],n_results=1)
            docs = results["documents"]
            distances = results["distances"]
            if docs and (1-distances[0][0])>similarity_limit:
                return docs[0][0], 1-distances[0][0] #return the first result, and the similarity (1-distance)
            

    def add(self, prompt_hash:UUID, prompt_embedding:List[float],result:str):
        
        self.collection.add(
            embeddings=[prompt_embedding],
            documents=[result],
            ids=[str(prompt_hash)],
        )
        self.collection._client.persist()
        self.has_data=True

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.has_data=False
        self.collection=self.client.create_collection(self.collection.name, self.collection.embedding_function)


class RemoteImpl(CacheImplBase):
    def __init__(self, promptwatch_context, cache_namespace_key:str =None,embed_func=None )-> None:
        self.promptwatch_context:PromptWatch=promptwatch_context
        self.promptwatch_context.tracking_project
        self.cache_namespace_key = f"{self.promptwatch_context.tracking_project or DEFAULT_CACHE_KEY}.{cache_namespace_key}"
        self.client = self.promptwatch_context.client

    
    def get(self, prompt_embedding:List[float], similarity_limit:float = 0.97)->  Tuple[str, float]:
        return self.client.get_from_cache(self.cache_namespace_key, prompt_embedding, min_similarity=similarity_limit)
    
    
    def add(self, prompt_hash:UUID, prompt_embedding:List[float],result:str):
        self.client.add_into_cache(self.cache_namespace_key, str(prompt_hash), prompt_embedding, result)

    def clear(self):
        self.client.clear_cache(self.cache_namespace_key)


def num_tokens_from_string(string: str, encoding_name: str="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
