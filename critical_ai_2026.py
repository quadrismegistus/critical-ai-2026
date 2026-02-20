"""
Minimal helpers for the Critical AI course.

This repository intentionally ships as a single importable module so it’s easy to
`pip install` and use from Colab notebooks.
"""
__version__ = "0.1.0"

DEFAULT_MODEL = "openai/gpt-4o"

import os
from dotenv import load_dotenv
import urllib.request
import json
from litellm import completion

def load_api_keys(url):
    urllib.request.urlretrieve(url, '.env')
    load_dotenv()

    # test API keys
    print(f'Anthropic API key set? = {"Yes" if os.getenv('ANTHROPIC_API_KEY') else "No"}')
    print(f'OpenAI API key set? = {"Yes" if os.getenv('OPENAI_API_KEY') else "No"}')
    print(f'Gemini API key set? = {"Yes" if os.getenv('GEMINI_API_KEY') else "No"}')
    print(f'DeepSeek API key set? = {"Yes" if os.getenv('DEEPSEEK_API_KEY') else "No"}')





# Define a generate_text function
def generate_text(
        user_prompt,                 # we can specify the user prompt (e.g. "Who are you?") as a string
        model=DEFAULT_MODEL,       # specify the model name
        system_prompt = "",          # specify the 'system prompt' (e.g. "You are a pirate") as a string
        verbose=True,                # whether to print out the response token by token
        temperature=None,            # how 'random' the output (0 = deterministic, 1 = very random)
        max_tokens=200,              # how many tokens to allow for the output
        **options                    # other options here: https://docs.litellm.ai/docs/completion/input
):

    # make the messages list
    messages = []

    # start with a system prompt if we have one
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # add the user prompt
    messages.append({"role": "user", "content": user_prompt})

    # start a list for the individual token responses
    tokens = []

    # force streaming
    options['stream']=True

    # create the response object
    response = completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **options
    )

    # loop through the returned tokens
    for token_obj in response:
        # get the token
        token = token_obj.choices[0].delta.content or ""

        # if verbose, print the token
        if verbose:
            print(token,end="",flush=True)

        # add this token to the list of tokens
        tokens.append(token)

    # return all tokens together as one string
    return "".join(tokens)




def generate_json(
    user_prompt,                 # we can specify the user prompt (e.g. "Who are you?") as a string
    model=DEFAULT_MODEL,       # specify the model name
    system_prompt = "",          # specify the 'system prompt' (e.g. "You are a pirate") as a string
    **options
):
    # get response
    response = generate_text(
        user_prompt=user_prompt,
        model=model,
        system_prompt=system_prompt,
        **options
    )

    # try to json parse it
    try:
        response = response.split("```json",1)[-1]
        response_l = response.split("```")
        response = response_l[1] if len(response_l) > 1 and response_l[1] else response_l[0]
        response_json = json.loads(response.strip())
        return response_json
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def truncate_text(text, max_lines):
    lines = text.split("\n")
    out = []
    nl = 0
    for i, line in enumerate(lines):
        out.append(line)
        if line.strip():
            nl += 1
        if nl > max_lines:
            break
        
    return "\n".join(out)
