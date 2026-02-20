from dotenv import load_dotenv
load_dotenv()                                                                                                           
import litellm             
                
for m in ['bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0', 'bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0', 'openai/gpt-5.2-2025-12-11']:                                                                          
    r = litellm.completion(
        model=m,       
        messages=[{'role': 'user', 'content': 'Rate this: Post says hello, comment says great post. Respond in JSON with fields: reasoning, responsiveness (1-5), information (1-5), category.'}],
        temperature=0,
        max_tokens=4096,                              
    )      
    print(f'{m}:')
    print(r.choices[0].message.content.strip()[:300])
    print()