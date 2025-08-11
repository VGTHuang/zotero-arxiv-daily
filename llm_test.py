from llm import get_llm, set_global_llm
import tiktoken
import argparse, os

# parser = argparse.ArgumentParser(description='Recommender system for academic papers')

# def add_argument(*args, **kwargs):
#     def get_env(key:str,default=None):
#         # handle environment variables generated at Workflow runtime
#         # Unset environment variables are passed as '', we should treat them as None
#         v = os.environ.get(key)
#         if v == '' or v is None:
#             return default
#         return v
#     parser.add_argument(*args, **kwargs)
#     arg_full_name = kwargs.get('dest',args[-1][2:])
#     env_name = arg_full_name.upper()
#     env_value = get_env(env_name)
#     if env_value is not None:
#         #convert env_value to the specified type
#         if kwargs.get('type') == bool:
#             env_value = env_value.lower() in ['true','1']
#         else:
#             env_value = kwargs.get('type')(env_value)
#         parser.set_defaults(**{arg_full_name:env_value})


# add_argument('--zotero_id', type=str, help='Zotero user ID')
# add_argument('--zotero_key', type=str, help='Zotero API key')
# add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
# add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
# add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
# add_argument('--arxiv_query', type=str, help='Arxiv search query')
# add_argument('--smtp_server', type=str, help='SMTP server')
# add_argument('--smtp_port', type=int, help='SMTP port')
# add_argument('--sender', type=str, help='Sender email address')
# add_argument('--receiver', type=str, help='Receiver email address')
# add_argument('--sender_password', type=str, help='Sender email password')
# add_argument(
#     "--use_llm_api",
#     type=bool,
#     help="Use OpenAI API to generate TLDR",
#     default=False,
# )
# add_argument(
#     "--openai_api_key",
#     type=str,
#     help="OpenAI API key",
#     default=None,
# )
# add_argument(
#     "--openai_api_base",
#     type=str,
#     help="OpenAI API base URL",
#     default="https://api.openai.com/v1",
# )
# add_argument(
#     "--model_name",
#     type=str,
#     help="LLM Model Name",
#     default="gpt-4o",
# )
# add_argument(
#     "--language",
#     type=str,
#     help="Language of TLDR",
#     default="English",
# )
# parser.add_argument('--debug', action='store_true', help='Debug mode')
# args = parser.parse_args()
# assert (
#     not args.use_llm_api or args.openai_api_key is not None
# )  # If use_llm_api is True, openai_api_key must be provided
# if args.debug:
#     logger.remove()
#     logger.add(sys.stdout, level="DEBUG")
#     logger.debug("Debug mode is on.")
# else:
#     logger.remove()
#     logger.add(sys.stdout, level="INFO")
    

set_global_llm(api_key="sk-xqdbjbdoobtlmqvpgdyvejnzwcjnytzvxwjyajthvwzllrmd",
               base_url="https://api.siliconflow.cn/v1", model="Qwen/Qwen3-8B", lang="Chinese")


llm = get_llm()
prompt = """Given the title, abstract, introduction and the conclusion (if any) of a paper in latex format, generate a one-sentence TLDR summary in __LANG__:

\\title{__TITLE__}
\\begin{abstract}__ABSTRACT__\\end{abstract}
__INTRODUCTION__
__CONCLUSION__
"""
prompt = prompt.replace('__LANG__', llm.lang)
prompt = prompt.replace('__TITLE__', "Multimodal Causal-Driven Representation Learning for Generalizable Medical Image Segmentation")
prompt = prompt.replace('__ABSTRACT__', " Vision-Language Models (VLMs), such as CLIP, have demonstrated remarkable zero-shot capabilities in various computer vision tasks. However, their application to medical imaging remains challenging due to the high variability and complexity of medical data. Specifically, medical images often exhibit significant domain shifts caused by various confounders, including equipment differences, procedure artifacts, and imaging modes, which can lead to poor generalization when models are applied to unseen domains. To address this limitation, we propose Multimodal CausalDriven Representation Learning (MCDRL), a novel framework that integrates causal inference with the VLM to tackle domain generalization in medical image segmentation. MCDRL is implemented in two steps: first, it leverages CLIP’s cross-modal capabilities to identify candidate lesion regions and construct a confounder dictionary through text prompts, specifically designed to represent domain-specific variations; second, it trains a causal intervention network that utilizes this dictionary to identify and eliminate the influence of these domain-specific variations while preserving the anatomical structural information critical for segmentation tasks. Extensive experiments demonstrate that MCDRL consistently outperforms competing methods, yielding superior segmentation accuracy and exhibiting robust generalizability. Our code is available at https://github.com/Xiaoqiovo/MCDRL.")

# use gpt-4o tokenizer for estimation
enc = tiktoken.encoding_for_model("gpt-4o")
prompt_tokens = enc.encode(prompt)
prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
prompt = enc.decode(prompt_tokens)

tldr = llm.generate(
    messages=[
        {
            "role": "system",
            "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
        },
        {"role": "user", "content": prompt},
    ]
)
print(tldr)