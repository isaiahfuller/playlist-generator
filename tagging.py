import os
from mutagen.mp3 import MP3, EasyMP3
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis

class Track:
    def __init__(self, file_name):
        """Check extension and get info"""
        try:
            if not os.path.exists(file_name):
                raise FileNotFoundError
            self.__f = self.__get_file(file_name)
        except FileNotFoundError:
            print("[Track] File not found")
        except:
            print("[Track] File is unsupported")
            self.__del__()

    def __del__(self):
        super()

    def __get_file(self,file_name) -> MP3 | FLAC | OggVorbis:
        name_split = file_name.split('.')
        extension = name_split[-1]
        match extension:
            case "flac":
                print(f"[Track] FLAC: {file_name}")
                return FLAC(file_name)
            case "mp3":
                print(f"[Track] MP3: {file_name}")
                return EasyMP3(file_name)
            case "ogg":
                print(f"[Track] OGG: {file_name}")
                return OggVorbis(file_name)
        print(f"[Track] Invalid: {file_name}")
        raise Exception()

    def get_tags(self):
        return self.__f

    def get_filtered_tags(self, opt):
        filtered = {key: value for key, value in self.__f.items() if opt in key}
        return filtered
