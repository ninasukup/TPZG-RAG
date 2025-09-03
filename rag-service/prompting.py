def build_rag_prompt(user_prompt: str, context_str: str) -> str:
    """
    Build an improved RAG prompt that can handle both specific questions and summarization tasks.
    """
    
    # Detect if this is a summarization request
    summary_keywords = ['summarize', 'summary', 'write', 'overview', 'describe', 'explain', 'tell me about']
    is_summary_request = any(keyword in user_prompt.lower() for keyword in summary_keywords)
    
    if is_summary_request:
        rag_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are an expert assistant for analyzing technical proposals and documents. "
            "Your task is to provide comprehensive, well-structured responses based on the provided documents. "
            "Use only the information from the documents provided - do not add outside knowledge.\n\n"
            
            "For summarization or overview requests:\n"
            "- Provide detailed, comprehensive responses that synthesize information across multiple documents\n"
            "- Include specific examples, technical details, and context when available\n"
            "- If the user requests a specific word count (e.g., '200 words'), aim to meet that length while staying informative\n"
            "- Organize your response logically with clear sections or paragraphs\n\n"
            
            "Structure your response as follows:\n"
            "1. **Executive Summary** – A brief overview addressing the main question\n"
            "2. **Detailed Analysis** – Comprehensive information organized by topic/document/timeline as appropriate\n"
            "3. **Key Technical Details** – Important specifications, features, or technical aspects\n"
            "4. **Additional Context** – Supporting information that adds value to the response\n"
            "5. **Sources Referenced** – List the documents used in your analysis\n\n"
            
            "If no relevant information is found, state: 'The provided documents do not contain information about [specific topic]. "
            "However, the documents do contain information about: [list what IS available]' and then summarize what you found.\n\n"
            
            f"Here are the most relevant documents:\n\n{context_str}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Request: {user_prompt}\n\n"
            "Provide a comprehensive, well-structured response based on the available documents."
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        rag_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are an expert assistant for analyzing technical proposals. "
            "Answer the user's question based on the provided documents. "
            "Use only information from the provided documents - do not add outside knowledge.\n\n"
            
            "If you cannot find a direct answer:\n"
            "- Look for related or partially relevant information\n"
            "- Explain what information IS available in the documents\n"
            "- Provide context about related topics found in the documents\n"
            "- Only use 'Not stated in the provided documents' as a last resort when absolutely no relevant information exists\n\n"
            
            "Structure your response as follows:\n"
            "1. **Direct Answer** – Answer the specific question if information is available\n"
            "2. **Supporting Details** – Additional context, specifications, or related information\n"
            "3. **Related Information** – Other relevant details found in the documents (if the direct answer is limited)\n"
            "4. **Sources** – List the most relevant source filenames used\n\n"
            
            f"Here are the most relevant documents:\n\n{context_str}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Question: {user_prompt}\n\n"
            "Provide a comprehensive response using the structured format above."
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    
    return rag_prompt


# Alternative: Single comprehensive prompt that handles both cases
def build_universal_rag_prompt(user_prompt: str, context_str: str) -> str:
    """
    Universal RAG prompt that adapts to both specific questions and summary requests.
    """
    
    rag_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert assistant for analyzing technical proposals and documents. "
        "Provide comprehensive, detailed responses based exclusively on the provided documents. "
        "Do not use outside knowledge or make assumptions beyond what's explicitly stated.\n\n"
        
        "**Response Guidelines:**\n"
        "- For specific questions: Provide direct answers with supporting details\n"
        "- For summary/overview requests: Provide comprehensive analysis with rich detail\n"
        "- If requesting specific length (e.g., '200 words'), aim to meet that target\n"
        "- Always look for related information even if the exact answer isn't found\n"
        "- Synthesize information across multiple documents when relevant\n"
        "- Include technical details, specifications, and context when available\n\n"
        
        "**When information is limited:**\n"
        "- Explain what information IS available in the documents\n"
        "- Provide relevant context from related topics in the documents\n"
        "- State clearly what specific information is missing\n"
        "- Only use 'Not stated in the provided documents' when absolutely no relevant information exists\n\n"
        
        "**Structure your response:**\n"
        "1. **Main Response** – Directly address the user's request with available information\n"
        "2. **Detailed Analysis** – Expand with supporting details, technical specifications, context\n"
        "3. **Additional Context** – Related information that adds value (if applicable)\n"
        "4. **Information Gaps** – Clearly state what specific information is not available (if relevant)\n"
        "5. **Sources** – List the primary source filenames referenced\n\n"
        
        f"Here are the most relevant documents:\n\n{context_str}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Request: {user_prompt}\n\n"
        "Provide a comprehensive, well-structured response based on the available documents."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    return rag_prompt