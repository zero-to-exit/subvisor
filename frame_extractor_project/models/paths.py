# Convert frame numbers to paths
from pathlib import Path
def frame_number_to_path(total_frame_dir: Path, frame_num: int) -> Path:
    """Convert frame number to file path (번호.jpg format)"""
    return total_frame_dir / f"{frame_num}.jpg"


#GEMINI FILE API
def gemini_file_API(client):
    print('My files:')
    for f in client.files.list():
        print(' ', f.name)


def del_all_files(client):
    for f in client.files.list():
        client.files.delete(name=f.name)
    print("All files deleted")
    return

def count_tokens(client, prompt, gemini_model = "gemini-2.5-flash-lite"):
    '''
    method to count the tokens used for the prompt. for free.
    '''
    # Count tokens using the new client method.
    total_tokens = client.models.count_tokens(
    model=gemini_model, contents=prompt
    )
    print("total_tokens: ", total_tokens)

    return total_tokens