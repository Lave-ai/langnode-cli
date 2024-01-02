ascii_art = """
   ______                      ___     _                              _
  |  ___ \  ____   ____       |   \   | |        _                   | |
  | |   \ |/ __ \ / __ \ _ __ | |\ \  | | __ _ _| |_ _   _  ___  __ _| |
  | |   | | /__\_| /__\_| '_ \| | \ \ | |/ _' |_  ._| | | |/ __|/ _' | |
  | |___/ | \____| \____| |_| | |  \ \| | |_| | | | | |_| | |  | |_| | |
  |______/ \_____|\_____|  __/|_|   \___|\__._| |_|  \__._|_|   \__._|_|
                        |_|                      
                        Add your product language
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

import inspect
import json
import wrapper

def list_classes_in_module():
   class_list = []
   for name, obj in inspect.getmembers(wrapper):
      if inspect.isclass(obj) and obj.__module__ == wrapper.__name__:
         class_list.append(obj)
   to_dict = []
   for c in class_list:
      if hasattr(c, "definition"):
         print(c)
         print(c.definition())
         to_dict.append(c.definition())

   
   with open("node_definitions.json", 'w') as f:
      json.dump(to_dict,f,ensure_ascii=False,indent=2)


if __name__ == "__main__":
   list_classes_in_module()