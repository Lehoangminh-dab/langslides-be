"""
Helper functions to access LLMs.
"""
import logging
import re
import sys
import urllib3
from typing import Tuple, Union, Optional, Any

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLLM, BaseChatModel


sys.path.append('..')

from global_config import GlobalConfig


LLM_PROVIDER_MODEL_REGEX = re.compile(r'\[(.*?)\](.*)')
OLLAMA_MODEL_REGEX = re.compile(r'[a-zA-Z0-9._:-]+$')
# 94 characters long, only containing alphanumeric characters, hyphens, and underscores
API_KEY_REGEX = re.compile(r'^[a-zA-Z0-9_-]{6,94}$')
HF_API_HEADERS = {'Authorization': f'Bearer {GlobalConfig.HUGGINGFACEHUB_API_TOKEN}'}
REQUEST_TIMEOUT = 35


logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.ERROR)

retries = Retry(
    total=5,
    backoff_factor=0.25,
    backoff_jitter=0.3,
    status_forcelist=[502, 503, 504],
    allowed_methods={'POST'},
)
adapter = HTTPAdapter(max_retries=retries)
http_session = requests.Session()
http_session.mount('https://', adapter)
http_session.mount('http://', adapter)


def get_provider_model(model_name: str, use_ollama: bool = True) -> Tuple[str, str]:
    """
    Parse the model name to determine provider and model.
    
    :param model_name: The model identifier, possibly with [provider] prefix.
    :param use_ollama: Whether to use Ollama as default provider.
    :return: Tuple of (provider, model_name).
    """
    # Handle OpenAI provider format [op]gpt-4o
    if model_name.startswith('[op]'):
        return 'openai', model_name[4:]
    
    # Default to Ollama
    return 'ollama', model_name


def is_valid_llm_provider_model(
        provider: str,
        model: str,
        api_key: str,
        azure_endpoint_url: str = '',
        azure_deployment_name: str = '',
        azure_api_version: str = '',
) -> bool:
    """
    Verify whether LLM settings are proper.
    This function does not verify whether `api_key` is correct. It only confirms that the key has
    at least five characters. Key verification is done when the LLM is created.

    :param provider: Name of the LLM provider.
    :param model: Name of the model.
    :param api_key: The API key or access token.
    :param azure_endpoint_url: Azure OpenAI endpoint URL.
    :param azure_deployment_name: Azure OpenAI deployment name.
    :param azure_api_version: Azure OpenAI API version.
    :return: `True` if the settings "look" OK; `False` otherwise.
    """

    if not provider or not model or provider not in GlobalConfig.VALID_PROVIDERS:
        return False

    if provider in [
        GlobalConfig.PROVIDER_GOOGLE_GEMINI,
        GlobalConfig.PROVIDER_COHERE,
        GlobalConfig.PROVIDER_TOGETHER_AI,
        GlobalConfig.PROVIDER_AZURE_OPENAI,
    ] and not api_key:
        return False

    if api_key and API_KEY_REGEX.match(api_key) is None:
        return False

    if provider == GlobalConfig.PROVIDER_AZURE_OPENAI:
        valid_url = urllib3.util.parse_url(azure_endpoint_url)
        all_status = all(
            [azure_api_version, azure_deployment_name, str(valid_url)]
        )
        return all_status

    return True


def get_langchain_llm(provider: str, model: str, max_new_tokens: int = 4096, api_key: Optional[str] = None) -> Any:
    """
    Get a LangChain LLM instance for the specified provider and model.
    
    :param provider: Provider ID ('ollama', 'openai').
    :param model: Model name.
    :param max_new_tokens: Maximum tokens to generate.
    :param api_key: API key for providers that require it (e.g., OpenAI).
    :return: LangChain LLM instance.
    """
    try:
        if provider == 'openai':
            logger.info(f"Creating OpenAI LLM with model: {model}")
            if not api_key:
                logger.error("No API key provided for OpenAI")
                return None
                
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                max_tokens=max_new_tokens,
                streaming=True
            )
        else:
            logger.info(f"Creating Ollama LLM with model: {model}")
            return ChatOllama(
                model=model,
                num_predict=max_new_tokens
            )
    except Exception as e:
        logger.error(f"Error creating LLM instance: {e}")
        return None


if __name__ == '__main__':
    inputs = [
        '[co]Cohere',
        '[hf]mistralai/Mistral-7B-Instruct-v0.2',
        '[gg]gemini-1.5-flash-002'
    ]

    for text in inputs:
        print(get_provider_model(text, use_ollama=False))
