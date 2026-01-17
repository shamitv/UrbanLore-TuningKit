import re
from typing import Dict, List, Any

def parse_world_bible_prompt(file_path: str) -> Dict[str, Any]:
    """
    Parses the World Bible markdown prompt into structured categories.
    
    Args:
        file_path: Path to the markdown file.
        
    Returns:
        Dict containing 'core_inputs' and a list of 'categories'.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract Core Inputs (simple extraction for now, can be refined)
    core_inputs_match = re.search(r'Core Inputs:\s+(.*?)\s+Please define', content, re.DOTALL)
    core_inputs = core_inputs_match.group(1).strip() if core_inputs_match else ""
    
    # Remove the preamble to focus on categories
    # Finding the start of the categories "I. The Physical World"
    start_match = re.search(r'I\. The Physical World', content)
    if not start_match:
        return {"core_inputs": core_inputs, "categories": []}
        
    main_content = content[start_match.start():]
    
    categories = []
    
    # Regex to find numbered categories like "1. Geography & Environment"
    # We capture:
    # 1. number
    # 2. title
    # 3. body (non-greedy until the next category start or end of string)
    # The lookahead is for the next number (e.g., "\n\n2. ") or Roman Numeral (e.g., "\n\nII. ")
    
    pattern = r'(?m)^(\d+)\.\s+(.*?)\n(.*?)(?=\n\d+\.|\n[IVX]+\.|$)'
    
    matches = re.finditer(pattern, main_content, re.DOTALL)
    
    for match in matches:
        num = match.group(1)
        name = match.group(2).strip()
        details_raw = match.group(3).strip()
        
        # Clean up the details (remove empty lines, etc)
        details = "\n".join([line.strip() for line in details_raw.split('\n') if line.strip()])
        
        categories.append({
            "id": num,
            "name": name,
            "prompt": details,
            "full_prompt": f"## {name}\n\n{details}"
        })
        
    return {
        "core_inputs": core_inputs,
        "categories": categories
    }

if __name__ == "__main__":
    # Test execution
    import sys
    try:
        path = sys.argv[1]
    except IndexError:
        path = r"d:\work\UrbanLore-TuningKit\docs\agents\sample_prompt.md"
        
    result = parse_world_bible_prompt(path)
    print(f"Found {len(result['categories'])} categories.")
    for cat in result['categories']:
        print(f"[{cat['id']}] {cat['name']}")
