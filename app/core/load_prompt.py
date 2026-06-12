from pathlib import Path
from app.utils.path_util import PROJECT_ROOT
from app.core.logger import logger  # Optional: for better logging

def load_prompt(name: str, **kwargs) -> str:
    """
    Loads a prompt template and renders variables into placeholders.

    :param name: The name of the prompt file (without the .prompt extension).
    :param **kwargs: Key-value pairs of variables to inject into the template 
                     (e.g., root_folder="Manual", image_content=("prev", "next")).
    :return: The final rendered prompt string.
    """
    # 1. Construct the prompt file path
    prompt_path = PROJECT_ROOT / 'prompts' / f'{name}.prompt'

    # 2. Validate file existence
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found at: {prompt_path.absolute()}")

    # 3. Read the raw text
    raw_prompt = prompt_path.read_text(encoding='utf-8')

    # 4. Render variables if provided; otherwise, return raw text
    if kwargs:
        rendered_prompt = raw_prompt.format(**kwargs)
        logger.debug(f"Prompt rendered successfully with variables: {list(kwargs.keys())}")
        return rendered_prompt
        
    return raw_prompt

if __name__ == '__main__':
    # Test: Injecting variables into the prompt template
    root_folder = "hl3070_user_manual"
    image_content = ("Header context", "Footer context")
    
    # Note: Variable names must match the placeholders in your .prompt file exactly
    final_prompt = load_prompt(
        name='image_summary',
        root_folder=root_folder,      # Maps to {root_folder}
        image_content=image_content   # Maps to {image_content[0]}, {image_content[1]}
    )
    
    print("✅ Final Rendered Prompt:")
    print(final_prompt)