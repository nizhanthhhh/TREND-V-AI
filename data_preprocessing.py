import pandas as pd
import numpy as np
import re
import unicodedata
import warnings
warnings.filterwarnings('ignore')

# Try to import emot, but handle if it's not available
try:
    import emot
    EMOT_AVAILABLE = True
except ImportError:
    EMOT_AVAILABLE = False
    print("Note: 'emot' library not found. Using fallback emoji removal method.")

def read_data(path: str) -> pd.DataFrame:
    """
    Reads a json file and returns a pandas dataframe.
    
    Args:
    path (str): Path to the json file.
    
    Returns:
    pd.DataFrame: A pandas dataframe containing the data.
    """
    try:
        df = pd.read_json(path)
        print(f"✓ Successfully loaded {path}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File '{path}' not found. Please make sure the data files exist.")
        raise
    except Exception as e:
        print(f"✗ Error loading {path}: {e}")
        raise

def replace_emojis_with_text(text: str) -> str:
    """
    This function replaces any emojis in a given text with their respective textual descriptions.
    Handles cases where the emot library fails.

    Args:
        text (str): Input text which may contain emojis.

    Returns:
        str: Output text with emojis replaced by their textual descriptions.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # If text is empty, return it
    if not text.strip():
        return text
    
    try:
        # Try using the emot library if available
        if EMOT_AVAILABLE:
            try:
                emot_obj = emot.emot()
                emoji_info = emot_obj.emoji(text)
                num_emojis = len(emoji_info["value"])
                
                if num_emojis > 0:
                    # Replace emojis with their meanings
                    for i in range(num_emojis):
                        if i < len(emoji_info["value"]) and i < len(emoji_info["mean"]):
                            text = text.replace(emoji_info["value"][i], f" {emoji_info['mean'][i]} ")
                    return ' '.join(text.split())
            except Exception as e:
                # If emot fails, fall through to fallback method
                pass
        
        # Fallback method: Remove emojis using Unicode categories
        try:
            # Method 1: Remove all emojis using Unicode categories
            cleaned = ''.join(char for char in text if not unicodedata.category(char).startswith('So'))
            
            # Method 2: Remove any remaining non-ASCII characters (except spaces)
            cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
            
            # Method 3: Clean up extra spaces
            cleaned = ' '.join(cleaned.split())
            
            return cleaned if cleaned.strip() else text
            
        except Exception as fallback_error:
            # Last resort: just return the text without emojis
            # Remove anything that's not a printable ASCII character
            return ''.join(char for char in text if ord(char) < 128 or char.isspace())
            
    except Exception as e:
        # If everything fails, return the original text
        print(f"Warning: Could not process text: {text[:50]}...")
        return text

def process_data(df: pd.DataFrame) -> tuple:
    """
    Processes the dataframe and returns a new dataframe.
    
    Args:
    df (pd.DataFrame): Dataframe to be processed.
    
    Returns:
    tuple: (processed dataframe, comments dataframe)
    """
    print("\n" + "="*50)
    print("PROCESSING DATA")
    print("="*50)
    
    # Selecting relevant columns
    required_columns = ["id", "type", "commentsCount", "likesCount", "latestComments", "images"]
    df = df[required_columns]
    print(f"✓ Selected {len(df)} rows")
    
    # Renaming columns
    new_columns = {
        "id": "id",
        "commentsCount": "n_comments",
        "likesCount": "n_likes",
        "latestComments": "comments",
        "images": "image"
    }
    df = df.rename(columns=new_columns)
    
    # Filtering out rows where type is "Video", likes count is -1.0 and image is not null
    df = df[(df["type"] != "Video")] 
    df = df[df["n_likes"] != -1.0]
    df = df[df["image"].notna()]
    print(f"✓ After filtering videos and invalid likes: {len(df)} rows")
    
    # Removing rows with no image
    df = df[df["image"].apply(len) > 0]
    print(f"✓ After removing rows with no images: {len(df)} rows")
    
    # Selecting the first image if there are multiple images
    df["image"] = df["image"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    
    # Extracting text from comments
    df["comments"] = df["comments"].apply(
        lambda x: [i["text"] for i in x if isinstance(i, dict) and "text" in i] 
        if isinstance(x, list) else []
    )
    
    # Resetting the id
    df.reset_index(drop=True, inplace=True)
    df["id"] = df.index + 1
    
    # Converting empty lists in comments to np.nan and creating a separate dataframe for comments
    df["comments"] = df["comments"].apply(lambda x: x if isinstance(x, list) and x else np.nan)
    df_comments = df.explode("comments")[["id", "comments"]]
    df_comments = df_comments.dropna(subset=["comments"])
    print(f"✓ Created {len(df_comments)} comments")
    
    # Replace emojis in comments with their text descriptions
    print("✓ Processing emojis in comments...")
    df_comments["comments"] = df_comments["comments"].apply(replace_emojis_with_text)
    
    # Remove any empty comments after processing
    df_comments = df_comments[df_comments["comments"].str.strip() != ""]
    print(f"✓ After cleaning comments: {len(df_comments)} comments")
    
    # Removing comments column from the original dataframe
    df = df.drop("comments", axis=1)
    
    return df, df_comments

def save_data(df: pd.DataFrame, df_comments: pd.DataFrame) -> None:
    """
    Saves the dataframe and comments dataframe as csv files.
    
    Args:
    df (pd.DataFrame): Dataframe to be saved.
    df_comments (pd.DataFrame): Comments dataframe to be saved.
    """
    print("\n" + "="*50)
    print("SAVING DATA")
    print("="*50)
    
    try:
        df.to_csv("data/posts.csv", index=False)
        print("✓ Saved: data/posts.csv")
        
        df_comments.to_csv("data/posts_comments.csv", index=False, sep=';')
        print("✓ Saved: data/posts_comments.csv")
        
        print(f"  - Posts: {len(df)} rows")
        print(f"  - Comments: {len(df_comments)} rows")
        print("="*50)
    except Exception as e:
        print(f"✗ Error saving data: {e}")
        raise
    
def main():
    """
    Main function to run the data preprocessing pipeline.
    """
    print("\n" + "="*50)
    print("FASHION TREND DATA PREPROCESSING")
    print("="*50)
    
    # Define paths
    path = "data/posts_1.json"
    path2 = "data/posts_2.json"
    
    try:
        # Read, process and save the data
        print("\n[1] Reading data files...")
        df_1 = read_data(path)
        df_2 = read_data(path2)
        
        print("\n[2] Concatenating data...")
        df = pd.concat([df_1, df_2], ignore_index=True)
        print(f"✓ Total rows: {len(df)}")
        
        print("\n[3] Processing data...")
        df, df_comments = process_data(df)
        
        print("\n[4] Saving data...")
        save_data(df, df_comments)
        
        print("\n" + "="*50)
        print("✓ PREPROCESSING COMPLETE!")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Missing data files!")
        print(f"  Please ensure that 'data/posts_1.json' and 'data/posts_2.json' exist.")
        print(f"  Error details: {e}")
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()