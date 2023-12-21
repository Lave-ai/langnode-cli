import typer
from rich import print


def main(name: str, lastname: str="", formal: bool= False):
    if formal:
        print(f"Good day Ms. {name} {lastname}.")
    else:
        ascii_art = """
   ______                      ___     _                              _
  |  ___ \  ____   ____       |   \   | |        _                   | |
  | |   \ |/ __ \ / __ \ _ __ | |\ \  | | __ _ _| |_ _   _  ___  __ _| |
  | |   | | /__\_| /__\_| '_ \| | \ \ | |/ _' |_  ._| | | |/ __|/ _' | |
  | |___/ | \____| \____| |_| | |  \ \| | (_| | | | | |_| | |  | (_| | |
  |______/ \_____)\_____|  __/|_|   \___|\__._| |_|  \__._|_|   \__._|_|
                        |_|                      
                 Life is too short, you need deepnatural
                               __         __
                              |  \       /  |
                              |   \_____/   |
                              |             |
                              |   ,     ,   |
                              |   .     .   |
                              |     _'_     |
                              |             |
                               \___________/
"""
        print(ascii_art)

        print(f"Hello {name} {lastname}")

if __name__ == "__main__":
    typer.run(main)