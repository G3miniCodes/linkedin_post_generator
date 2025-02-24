import json
import re
from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


def clean_text(text):
    """Removes unsupported Unicode characters from the text."""
    return text.encode("utf-8", "ignore").decode("utf-8")


def process_posts(raw_file_path, processed_file_path=None):
    """Processes LinkedIn posts, extracts metadata, and unifies tags."""
    with open(raw_file_path, encoding="utf-8", errors="replace") as file:
        posts = json.load(file)

    enriched_posts = []
    for post in posts:
        post['text'] = clean_text(post['text'])  
        metadata = extract_metadata(post['text'])
        post_with_metadata = post | metadata  # Merge metadata 
        enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)
    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags.get(tag, tag) for tag in current_tags}  # Fallback to original tag if not found
        post['tags'] = list(new_tags)

    with open(processed_file_path, "w", encoding="utf-8", errors="ignore") as outfile:
        json.dump(enriched_posts, outfile, indent=4, ensure_ascii=False)


def extract_metadata(post):
    """Extracts metadata like line count, language, and tags from the post using LLM."""
    template = '''
Extract the following details from the LinkedIn post:
- line_count
- language
- tags

Provide ONLY valid JSON output, with no explanations.

Example output: {{"line_count": 3, "language": "Hinglish", "tags": ["Influencer", "Organic Growth"]}}

Rules:
1. Return a valid JSON. No preamble.
2. JSON object should have exactly three keys: line_count, language, and tags.
3. tags should be an array of at most two text tags.
4. Language should be "English" or "Hinglish" (Hinglish = Hindi + English).

Here is the actual post:
{post}
'''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": post})

    try:
        json_parser = JsonOutputParser()
        return json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse metadata.")


def get_unified_tags(posts_with_metadata):
    """Unifies similar tags into a standardized format."""
    unique_tags = set()
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])  

    unique_tags_list = ', '.join(unique_tags)

    template = '''
    I will give you a list of tags. Your task is to unify them into a meaningful format.

    **Rules:**
    1. Merge similar tags into a single, standard tag.
       Examples:
       - "Jobseekers", "Job Hunting" → "Job Search"
       - "Motivation", "Inspiration", "Drive" → "Motivation"
       - "Personal Growth", "Personal Development", "Self Improvement" → "Self Improvement"
       - "Scam Alert", "Job Scam" → "Scams"
    2. Output **only a valid JSON object** mapping original tags to unified tags.
    3. No explanations, preambles, or additional formatting.
    
    **Input tags:** {tags}

    **Expected output format:**
    ```json
    {{"Jobseekers": "Job Search", "Motivation": "Motivation", "Job Scam": "Scams"}}
    ```
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": unique_tags_list})

    try:
        json_parser = JsonOutputParser()
        return json_parser.parse(response.content)
    except OutputParserException:
        print("Error: LLM response did not return valid JSON. Here is the raw response:\n", response.content)
        return {tag: tag for tag in unique_tags}  


if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed_posts.json")
