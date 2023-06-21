import os
import sqlite3

def idToPath(id):
    # global dataset
    id_str = (str(id)[::-1])[:-1] if len(str(id)) % 2 == 1 else str(id)[::-1]
    return '/{0}'.format('/'.join([id_str[i:i + 2] for i in range(0, len(id_str), 2)]))

class DataAcquirer:
    def __init__(self, databasePath, datasetPath):
        self.database = databasePath
        self.dataset = datasetPath

    def bootup(self):
        # connect to the database
        self.database = sqlite3.connect(self.database)

        # create a cursor object
        self.cursor = self.database.cursor()
    def filter(self, table, field, value):
        self.cursor.execute("SELECT id, file_name, size FROM binaries WHERE platform='x86'")
        rows = self.cursor.fetchall()
        return rows

    def getFileInfo(self, rows):
        file_info = []
        for id, file_name, file_size in rows:
            path = (self.dataset + idToPath(id) + '/' + file_name).replace('//', '/')
            file_info.append((id, file_name, file_size, path))
        return file_info
def main():
    dataThing = DataAcquirer('H:/Datasets/Bigger_dataset/data.sqlite', 'H:/Datasets/Bigger_dataset/dataset/')
    dataThing.bootup()
    rows = dataThing.filter('binaries', 'platform', 'x86')
    file_info = dataThing.getFileInfo(rows)

    mismatches = []
    notFound = []
    # check the file size for each file
    for item in file_info:
        id, file_name, file_size, path = item
        try:
            sizeInKb = os.path.getsize(path) / 1024
            actual_size = int(sizeInKb)
            if actual_size != file_size:
                # remove item from file_info
                file_info.remove(item)
                mismatches.append((id, file_name, file_size, path, actual_size))

        except FileNotFoundError:
            # remove item from file_info
            file_info.remove(item)
            notFound.append((id, file_name, file_size, path))

    print(f"Number of files not found: {len(notFound)}")
    print(f"Number of files with mismatched sizes: {len(mismatches)}")
    print(f"Number of files with correct sizes: {len(file_info)}")
#     print names, ids, paths, sizes for notFound
    print("Not found:")
    for item in notFound:
        print(item)



if __name__ == '__main__':
    main()

