This project uses Tensorflow and models from Essentia to scan music files, categorizing genres and moods, and gives you a playlist of songs similar to one you choose.

Installation:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Usage:  
`-b or --base {dir}` - folder with songs (supports mp3, flac, and ogg files)  
`-t or --tags` - saves song's tags to the db (currently unused)  
`-s or --sonic` - runs the tensorflow scanner for genre/mood classifiers  
`-pl or --playlist` {song path} - generates playlist based on the given song  

Note: The sonic scanner can take a ***VERY*** long time. On my library of ~50000 songs, it took multiple days to finish.
