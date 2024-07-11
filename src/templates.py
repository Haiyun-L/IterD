
EX_template = '''We will now perform an Aspect-Based Sentiment Analysis task. In this task, you are required to:
- Identify the aspects mentioned in the text 
- Determine the sentiment polarity toward each aspect (positive, neutral, or negative)
- Output format: [aspect, sentiment_polarity]
{example}
Now, complete the aspect extraction task for the text below:
Input: "{input}"
Output: '''

ET_template = '''We will now perform an Aspect-Based Sentiment Analysis task, in this task, Expansion of homonyms or synonyms for a given aspectual word
Generation of 5-10 cognate or synonymous aspect words for an aspect word
example:
input:{example_in}
output:{example_out}
Now, complete the aspect extend task for the text below:
input: {input} 
output:'''

Eval_filter = '''You need to perform a task of sentiment judgment and domain judgment, the task requirements are shown below:
- Determine whether the potential sentiment hidden in the sentence by aspect is positive,
negative, or neutral based on the context given in the sentence.
- Avoid confusing the neutral sentiment of the aspect with a positive or negative sentiment.
- Is this sentence related to {domain} ? If so, output “Y”; otherwise, output “N”.
- Here are some examples of how aspect represents the sentiment in a sentence for your
reference:
example-input:{[aspect, sentiment] }
example-output:{[sentence, #aspect, sentiment]}
Now, please complete the task for the following input:
- input format: sentence, #aspect
- output format: sentiment; Y(N)
Input: {input}
Output:
 '''

Eval_score = '''{example}
You are a psycholinguist who analyses sentiment and scores the above sentences in the
following three areas:
1. Possessing complex syntactic structures, such as inverted sentences, imperative sen-
tences, sentences with inflections, and sentences beginning with multiple combinations of
adverbs, nouns, and subjects, the more complex the higher the score.
2. With a rich vocabulary, the richer the score, the higher the score.
3. User comments that match real-life scenarios, the more they match, the higher the score.
Please give a score of 1-10 from each aspect accurately, and finally output a comprehensive average score selection of the highest-scoring sentences, the requirements of the output format are as follows:
[syntactic-structure: score; vocabulary-richness: score; real-scenario-conformity: score; comprehensive score: score]
Please output in decimal form:'''

ITAT_template = '''We would like you to complete a sentence generation task, , and we will tell you how to
generate appropriate sentences. Please follow these requirements:
-Teaching analysis – analyzing the given aspect and sentiment:
- Specify the sentiment of the aspect in the generated sample.
- Domain of sample generation: {domain}
- Generate a sentence containing a given aspect, clarify the meaning of the aspect, and generate sentences corresponding to the polarity of the sentiment.
- The generated sentence must be in length within {length} words.
- Generated sentences can contain only one period at a time and the sentence should not consist of an unspecified aspect.
- examples:
Input: {example-input}
Output: {example-input}
Now, complete this task in a natural human-like manner and generate only one sentence:
Input: {input}
Output:'''
