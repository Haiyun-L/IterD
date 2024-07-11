import openai
import time
import warnings
from tenacity import (retry,stop_after_attempt,wait_random_exponential)

warnings.filterwarnings("ignore")
openai.api_base = ""
openai.api_key = ""

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def invoke_gpt_turbo_generate(prompt, temperature=1.0):
    retry_count = 20
    retry_interval = 1
    model_engine = "gpt-3.5-turbo-0613"
    for _ in range(retry_count):
        try:
            completion = openai.ChatCompletion.create(model=model_engine,
                                                      messages=[{"role": "system", "content": "You are a critic without feelings who can generate comments on specified aspect and sentiment. "},
                                                                {"role": "user", "content": prompt}], temperature=temperature
                                                      )
            return completion.choices[0].message.content
        except openai.error.RateLimitError as e:
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
        except TimeoutError:
            print("Timeout：", prompt)
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
        except Exception as e:
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
    return prompt

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def invoke_gpt_turbo(prompt, temperature=0, top_p=0.7):
    retry_count = 20
    retry_interval = 1
    model_engine = "gpt-3.5-turbo"
    for _ in range(retry_count):
        try:
            completion = openai.ChatCompletion.create(model=model_engine,
                                                      messages=[{"role": "system","content": "You are an AI assistant specializing in linguistics and sentiment analysis. "},
                                                                {"role": "user", "content": prompt}],temperature=temperature,top_p=top_p)
            return completion.choices[0].message.content
        except openai.error.RateLimitError as e:
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
        except TimeoutError:
            print("Timeout：", prompt)
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
        except Exception as e:
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
    return prompt
