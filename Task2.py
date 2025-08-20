import pandas as pd
import google.generativeai as genai
import os
import time
from typing import Optional
import sys
from dotenv import load_dotenv

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

class GameDataEnricher:
    def __init__(self, api_key: Optional[str] = None):
      
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "API key not found. Please provide an API key or set GOOGLE_API_KEY environment variable.\n"
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        # CONFIGURE THE API
        genai.configure(api_key=self.api_key)

        # USING GEMINI-1.5-FLASH WHICH IS AVAILABLE AND EFFICIENT
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.api_delay = 5.0  # SECONDS BETWEEN API CALLS
        
    def classify_genre(self, game_title: str) -> str:
      
        prompt = f"""
        Classify the video game "{game_title}" into ONE single-word genre.
        Choose from: Action, RPG, Sports, Strategy, Simulation, Adventure, 
        Puzzle, Racing, Fighting, Horror, Platformer, Shooter, MMORPG, Sandbox, Card
        
        Respond with ONLY the genre word, nothing else.
        
        Game: {game_title}
        Genre:
        """
        
        try:
            response = self.model.generate_content(prompt)
            genre = response.text.strip()
            
            if ' ' in genre:
                genre = genre.split()[0]
            
            return genre
        except Exception as e:
            print(f"Error classifying genre for {game_title}: {e}")
            return "Unknown"
    
    def generate_description(self, game_title: str) -> str:
        
        prompt = f"""
        Write a short description for the video game "{game_title}".
        The description must be under 30 words.
        Be concise and focus on the core gameplay and unique features.
        Do not include the game title in the description.
        
        Game: {game_title}
        Description:
        """
        
        try:
            response = self.model.generate_content(prompt)
            description = response.text.strip()
            
            words = description.split()
            if len(words) > 30:
                description = ' '.join(words[:29]) + '.'
            
            return description
        except Exception as e:
            print(f"Error generating description for {game_title}: {e}")
            return "A video game experience."
    
    def determine_player_mode(self, game_title: str) -> str:

        prompt = f"""
        Determine the player mode for the video game "{game_title}".
        
        Respond with ONLY ONE of these three options:
        - Singleplayer (if the game is primarily single-player only)
        - Multiplayer (if the game is primarily multiplayer only)
        - Both (if the game supports both single-player and multiplayer modes)
        
        Game: {game_title}
        Player Mode:
        """
        
        try:
            response = self.model.generate_content(prompt)
            mode = response.text.strip()
            
            valid_modes = ["Singleplayer", "Multiplayer", "Both"]
            
            if "single" in mode.lower():
                return "Singleplayer"
            elif "multi" in mode.lower() and "both" not in mode.lower():
                return "Multiplayer"
            elif "both" in mode.lower():
                return "Both"
            
            if mode in valid_modes:
                return mode
            
            return "Both"
            
        except Exception as e:
            print(f"Error determining player mode for {game_title}: {e}")
            return "Both"
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        # CREATE NEW COLUMNS
        genres = []
        descriptions = []
        player_modes = []
        
        total_games = len(df)
        
        print(f"Processing {total_games} games...")
        print("-" * 50)
        
        for idx, row in df.iterrows():
            game_title = row['game_title']
            print(f"Processing [{idx+1}/{total_games}]: {game_title}")
            
            # GET GENRE
            genre = self.classify_genre(game_title)
            genres.append(genre)
            print(f"  Genre: {genre}")
            
            # ADD DELAY TO AVOID RATE LIMITING
            time.sleep(self.api_delay)
            
            # GET DESCRIPTION
            description = self.generate_description(game_title)
            descriptions.append(description)
            print(f"  Description: {description[:50]}...")
            
            # ADD DELAY TO AVOID RATE LIMITING
            time.sleep(self.api_delay)
            
            # GET PLAYER MODE
            player_mode = self.determine_player_mode(game_title)
            player_modes.append(player_mode)
            print(f"  Player Mode: {player_mode}")
            
            # ADD DELAY BETWEEN GAMES
            time.sleep(self.api_delay)
            print()
        
        # ADD NEW COLUMNS TO DATAFRAME
        df['genre'] = genres
        df['short_description'] = descriptions
        df['player_mode'] = player_modes
        
        return df


def main():

    # FILE PATHS
    input_file = "Game_Thumbnail.csv"
    output_file = "Game_Thumbnail_New.csv"
    
    print("=" * 60)
    print("Video Game Data Enhancement Script")
    print("=" * 60)
    print()
    
    # CHECK IF INPUT FILE EXISTS
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please ensure the Game_Thumbnail.csv file is in the same directory.")
        sys.exit(1)
    
    # LOAD THE CSV FILE
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} games.")
        print()
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    # INITIALIZE THE ENRICHER
    try:
        enricher = GameDataEnricher()
        print("Google AI Studio API configured successfully.")
        print()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during initialization: {e}")
        sys.exit(1)
    
    # PROCESS THE DATAFRAME
    try:
        enhanced_df = enricher.process_dataframe(df)
        print("=" * 60)
        print("Processing complete!")
        print()
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)
    
    # SAVE THE ENHANCED DATAFRAME
    try:
        enhanced_df.to_csv(output_file, index=False)
        print(f"Enhanced data saved to: {output_file}")
        print()
        
        # DISPLAY SAMPLE RESULTS
        print("Sample of enhanced data:")
        print("-" * 60)
        print(enhanced_df[['game_title', 'genre', 'player_mode']].head())
        print()
        print("First game description sample:")
        print(f"Game: {enhanced_df.iloc[0]['game_title']}")
        print(f"Description: {enhanced_df.iloc[0]['short_description']}")
        
    except Exception as e:
        print(f"Error saving enhanced data: {e}")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("Script execution completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()