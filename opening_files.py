import os
import re
from collections import defaultdict

class CepheidFileGrouper:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.pattern = re.compile(r"Cepheids_(\d+)_00")
        self.groups = defaultdict(list)

    def discover_files(self):
        for root, _, files in os.walk(self.base_dir):
            for fname in files:
                if fname.lower().endswith(".fits"):
                    match = self.pattern.search(fname)
                    if match:
                        cepheid_number = match.group(1)   # e.g., "7"
                        fullpath = os.path.join(root, fname)
                        self.groups[cepheid_number].append(fullpath)

        # Sort lists so you keep chronological order
        for key in self.groups:
            self.groups[key].sort()

        return dict(self.groups)