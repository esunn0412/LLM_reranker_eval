"""
Prompt Templates for LLM-based Reranking
Based on RankGPT methodology
"""


def create_rankgpt_messages(query, passages):
    """
    Create RankGPT-style chat messages
    
    Args:
        query: Search query string
        passages: List of passage texts
    
    Returns:
        List of chat messages ready for LLM
    """
    num_passages = len(passages)
    
    messages = [
        {
            'role': 'system',
            'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
        },
        {
            'role': 'user',
            'content': f"I will provide you with {num_passages} passages, each indicated by number identifier []. "
                      f"Rank the passages based on their relevance to query: {query}."
        },
        {
            'role': 'assistant',
            'content': 'Okay, please provide the passages.'
        }
    ]
    
    # Add each passage with acknowledgment
    for i, passage in enumerate(passages):
        messages.append({
            'role': 'user',
            'content': f'[{i}] {passage}'
        })
        messages.append({
            'role': 'assistant',
            'content': f'Received passage [{i}].'
        })
    
    # Post ranking instruction
    messages.append({
        'role': 'user',
        'content': (
            f"Search Query: {query}. \n"
            f"Rank the {num_passages} passages above based on their relevance to the search query. "
            f"The passages should be listed in descending order using identifiers. "
            f"The most relevant passages should be listed first. "
            f"The output format should be the ranking as a list of identifiers, e.g., [3, 1, 2]. "
            f"Only respond with the ranking results, do not say any word or explain."
        )
    })
    
    return messages
